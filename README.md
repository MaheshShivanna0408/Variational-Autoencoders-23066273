# Variational Autoencoders: Learning to Generate with Uncertainty

A complete, step-by-step Jupyter notebook tutorial on Variational Autoencoders (VAEs) trained on FashionMNIST — covering the reparameterisation trick, ELBO loss, latent space visualisation, image generation, and class interpolation.

---

## What This Notebook Covers

| Step | Topic |
|------|-------|
| 0 | Install and import dependencies |
| 1 | Load and visualise FashionMNIST |
| 2 | Build the VAE architecture (encoder, reparameterisation, decoder) |
| 3 | Define the ELBO loss (reconstruction + KL divergence) |
| 4 | Train the VAE for 30 epochs |
| 5 | Plot training curves (reconstruction vs KL loss) |
| 6 | Visualise the 2D latent space by class |
| 7 | Compare original images with reconstructions |
| 8 | Generate images by traversing the latent space |
| 9 | Interpolate smoothly between two classes |
| 10 | Key takeaways |

Three central questions are answered throughout:

1. **Why does the reparameterisation trick exist?** — and how it makes backpropagation through a sampling step possible
2. **What does the latent space actually learn?** — class clusters emerge with no labels during training
3. **What is the reconstruction vs generation trade-off?** — the tension at the heart of VAEs

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/vae-tutorial.git
cd vae-tutorial
```

### 2. Install dependencies

Python 3.8+ is required. Install all packages with:

```bash
pip install torch torchvision matplotlib numpy
```

FashionMNIST is downloaded automatically by `torchvision` on first run. No manual dataset setup is needed.

### 3. Open the notebook

```bash
jupyter notebook vae_tutorial.ipynb
```

Run cells from top to bottom. All cells are self-contained and run in order.

---

## Dataset

**FashionMNIST** (Xiao et al., 2017) — 70,000 greyscale 28×28 images across 10 clothing categories.

| Label | Class |
|-------|-------|
| 0 | T-shirt / Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

FashionMNIST was chosen over standard MNIST because its 10 classes are more visually diverse, making the VAE's learned latent structure more informative and the tutorial more meaningful.

---

## Model Architecture

A fully connected VAE operating on flattened 784-dimensional images.

```
Encoder:  784 → 512 → 256 → (μ, log σ²)   — each of size latent_dim
Decoder:  latent_dim → 256 → 512 → 784
```

The encoder outputs a **mean** (μ) and **log-variance** (log σ²) rather than a single point. The reparameterisation trick samples `z = μ + σ * ε` where `ε ~ N(0, I)`, keeping gradients flowing through the stochastic node.

Log-variance is predicted (rather than variance directly) so that positivity is guaranteed without a constrained activation function, and for numerical stability.

---

## Loss Function

The negative ELBO, minimised during training:

```
L = BCE(x, x̂)  +  β · KL[ q(z|x) || p(z) ]
```

| Term | Role |
|------|------|
| Reconstruction loss (BCE) | Forces the decoder to accurately rebuild the input |
| KL divergence | Regularises the latent space to stay close to N(0, I) |
| β (beta) | Trade-off weight — set to 1.0 (standard VAE) |

The KL term has a closed-form solution for Gaussian distributions, computed analytically rather than via sampling.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 30 |
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Latent dimension | 2 |
| Random seed | 42 |
| Device | CUDA if available, else CPU |

---

## Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Sample images from FashionMNIST (2×5 grid, one per class) |
| Fig 2 | Training curves — total loss, reconstruction loss, and KL divergence over 30 epochs |
| Fig 3 | 2D latent space scatter plot — all test images coloured by class label |
| Fig 4 | Reconstruction quality — 10 test images shown alongside their VAE reconstructions |
| Fig 5 | Latent space traversal — 20×20 grid of decoded images sampled across the latent space |
| Bonus | Class interpolation — smooth morphing between a T-shirt and an Ankle boot |

---

## Key Concepts Demonstrated

| Concept | What the notebook shows |
|---------|------------------------|
| Reparameterisation trick | Separates randomness (ε) from learnable parameters (μ, σ), enabling gradient flow through the sampling step |
| ELBO loss | The two-term loss creates the tension that makes the latent space both accurate and structured |
| Unsupervised structure | Class clusters emerge in the latent space with no class labels used during training |
| Generation | Any point sampled from N(0,1) decodes to a plausible garment — impossible with a standard autoencoder |
| Interpolation | Smooth morphing between classes confirms the latent space is continuous and well-behaved |

---

## Accessibility

All figures use the **Wong (2011) 8-colour colourblind-safe palette**, distinguishable under deuteranopia, protanopia, and tritanopia. Figure DPI is set to 120 throughout for crisp rendering in notebooks.

---

## References

1. Kingma, D. P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR 2014. https://arxiv.org/abs/1312.6114
2. Xiao, H., Rasul, K. & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* https://arxiv.org/abs/1708.07747
3. Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017.
4. Doersch, C. (2016). *Tutorial on Variational Autoencoders.* https://arxiv.org/abs/1606.05908
5. Wong, B. (2011). *Color blindness.* Nature Methods, 8(6), 441. https://doi.org/10.1038/nmeth.1618
6. PyTorch Documentation. https://pytorch.org/docs/stable/index.html

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

