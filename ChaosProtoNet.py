import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==================== Device Management ====================
def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = torch_directml.device_name(0)
        logging.info(f"Using DirectML Device: {device_name}")
        return device
    except ImportError:
        logging.warning("torch_directml not found. Falling back to CPU.")
    except Exception as e:
        logging.warning(f"Failed to initialize DirectML ({e}). Falling back to CPU.")
    
    device = torch.device("cpu")
    logging.info("Using CPU")
    return device

device = get_device()

# ==================== CE-ProtoNet Modules ====================
class LogisticChaos(nn.Module):
    """
    Logistic Chaos Module: Injects dynamic, deterministic chaos noise 
    during the training phase to improve embedding generalization.
    """
    def __init__(self, intensity=0.04, r=3.99):
        super().__init__()
        self.intensity = intensity
        self.r = r
        
    def forward(self, x):
        if not self.training:
            return x
        noise = torch.rand_like(x)
        for _ in range(8):
            noise = self.r * noise * (1 - noise)
        noise = (noise - 0.5) * 2 * self.intensity
        return x + noise

class MedicalProtoNet(nn.Module):
    """
    Backbone network for CE-ProtoNet. 
    Uses ResNet-18 with L2 normalization for cosine similarity metric learning.
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.scale = nn.Parameter(torch.tensor(20.0))

        # Unfreeze last two blocks for fine-tuning
        for name, param in resnet.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        return F.normalize(z, dim=1)

# ==================== Episodic Task Sampler ====================
class MedicalTaskSampler:
    """
    Few-Shot Episodic Sampler for N-way K-shot classification.
    """
    def __init__(self, targets, n_way=4, k_shot=5, query=15, episodes=1000):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query = query
        self.episodes = episodes
        targets = np.array(targets)
        self.classes = np.unique(targets)
        self.idx_map = {c: np.where(targets == c)[0] for c in self.classes}

    def __len__(self): 
        return self.episodes
        
    def __iter__(self):
        for _ in range(self.episodes):
            batch = []
            classes = np.random.choice(self.classes, self.n_way, replace=False)
            for c in classes:
                idxs = self.idx_map[c]
                replace = len(idxs) < (self.k_shot + self.query)
                chosen = np.random.choice(idxs, self.k_shot + self.query, replace=replace)
                batch.append(chosen)
            yield np.concatenate(batch)

# ==================== Training & Validation ====================
def train_epoch(model, chaos, loader, opt, y_query, n_way, k_shot, query):
    model.train()
    total_loss = total_acc = 0
    for data, _ in tqdm(loader, desc="Training", leave=False):
        # Format: (N_way, K_shot + Query, Channels, Height, Width)
        data = data.view(n_way, k_shot + query, 3, 224, 224)
        
        # CPU contiguous slice before DirectML transfer for VRAM efficiency
        support = data[:, :k_shot].contiguous().view(-1, 3, 224, 224).to(device)
        q_data = data[:, k_shot:].contiguous().view(-1, 3, 224, 224).to(device)

        z_s = model(support)
        z_q = model(q_data)
        z_s = chaos(z_s)  # Inject chaos only on support embeddings
        proto = z_s.view(n_way, k_shot, -1).mean(1)  # Compute class prototypes

        sim = torch.mm(z_q, proto.t())
        loss = F.cross_entropy(sim * model.scale, y_query)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_acc += (sim.argmax(1) == y_query).float().mean().item()

    return total_loss / len(loader), total_acc / len(loader)

@torch.no_grad()
def validate(model, loader, y_query, n_way, k_shot, query):
    model.eval()
    accs = []
    for data, _ in loader:
        data = data.view(n_way, k_shot + query, 3, 224, 224)
        support = data[:, :k_shot].contiguous().view(-1, 3, 224, 224).to(device)
        q_data = data[:, k_shot:].contiguous().view(-1, 3, 224, 224).to(device)

        proto = model(support).view(n_way, k_shot, -1).mean(1)
        sim = torch.mm(model(q_data), proto.t())
        accs.append((sim.argmax(1) == y_query).float().mean().item())
    return np.mean(accs)

# ==================== MAIN CLI ====================
def main():
    parser = argparse.ArgumentParser(description="CE-ProtoNet: Chaos-Enhanced Prototypical Networks")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory containing 'train' and 'test' folders")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--episodes_train", type=int, default=500, help="Number of episodes per training epoch")
    parser.add_argument("--episodes_val", type=int, default=600, help="Number of episodes per validation epoch")
    parser.add_argument("--n_way", type=int, default=4, help="N-way classification (number of classes per episode)")
    parser.add_argument("--k_shot", type=int, default=5, help="K-shot (number of support samples per class)")
    parser.add_argument("--query", type=int, default=15, help="Number of query samples per class")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--chaos_intensity", type=float, default=0.18, help="Intensity parameter for the Logistic Chaos module")
    parser.add_argument("--output_model", type=str, default="ce_protonet_best.pth", help="Filename to save the best model weights")
    
    args = parser.parse_args()

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(args.data_path):
        logging.error(f"Dataset not found at {args.data_path}. Please check the --data_path argument.")
        return

    logging.info("Loading dataset...")
    train_ds = ImageFolder(os.path.join(args.data_path, "train"), train_tf)
    val_ds   = ImageFolder(os.path.join(args.data_path, "test"),  val_tf)

    train_loader = DataLoader(
        train_ds, 
        batch_sampler=MedicalTaskSampler(train_ds.targets, n_way=args.n_way, k_shot=args.k_shot, query=args.query, episodes=args.episodes_train)
    )
    val_loader = DataLoader(
        val_ds,   
        batch_sampler=MedicalTaskSampler(val_ds.targets, n_way=args.n_way, k_shot=args.k_shot, query=args.query, episodes=args.episodes_val)
    )

    model = MedicalProtoNet().to(device)
    chaos = LogisticChaos(intensity=args.chaos_intensity).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Pre-compute target labels on CPU first to avoid DML operator issues, then move to DML
    y_query = torch.arange(args.n_way).repeat_interleave(args.query).to(device)

    best = 0.0
    logging.info(f"Starting CE-ProtoNet Training ({args.n_way}-way {args.k_shot}-shot)...\n")
    
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, chaos, train_loader, opt, y_query, args.n_way, args.k_shot, args.query)
        val_acc = validate(model, val_loader, y_query, args.n_way, args.k_shot, args.query)
        sched.step()

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Train: {acc*100:5.2f}% | Val: {val_acc*100:5.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.output_model)
            print(f"[*] NEW BEST MODEL SAVED: {best*100:.2f}%")

    print(f"\nFINAL BEST ACCURACY: {best*100:.2f}%")
    print(f"Model successfully saved to: {args.output_model}")

if __name__ == "__main__":
    main()
