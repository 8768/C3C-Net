# C3C-Net
Understanding the distribution of crowd flow is crucial for optimizing intelligent transportation services such as ride-hailing and sharing-bike scheduling. A key element for these applications is the accurate and reliable prediction of origin-destination (OD) flows, which provide a detailed profile of how and when passengers enter and exit the service. Despite the interest in traffic prediction in recent years, OD flow estimation remains a challenging task for deep spatiotemporal networks due to the two unique characteristics: (i) City-wide OD human movements are quite sparse. OD flows between many region pairs could be zero over a long period, the data sparsity and representation is a problem for deep models, (ii) OD spatial relations are timesensitive and large-scale, and can affect region-level similarity, attractiveness, and pair-level correlation. To solve these issues, we propose a context-aware 3D criss-cross fusion network for city-wide OD crowd flow prediction (C3CNet). First, we design a novel two-stream representation framework, where a contextual-embedded pair stream and a geographical-based region stream maintain both OD and in/outflow distribution from generation/attraction and transaction perspectives while preserving dynamic contextual factors. The information involved for OD demands as data augmentation can help to solve the sparsity problem adaptively in the learning process. Then, the C3C-Net consists of several components during the learning process, such as 3D convolutional module and residual mechanism. In particular, a 3D criss-cross attention module to learn global multi-view spatiotemporal dependencies and heterogeneity simultaneously with high computing efficiency. We further propose a stream fusion attention module to enhance the model performance by combining multi-scale region and pair-level latent features.Evaluation on real-world crowd flow data demonstrates that our C3C-Net method outperforms existing state-of-the-art methods in terms of prediction accuracy.
<img width="564" height="188" alt="image" src="https://github.com/user-attachments/assets/55e17ad1-bb89-4a8b-88a9-d3098b2ed165" />
How to Contribute (Project Health)

We welcome all kinds of contributions, including:

Submitting Pull Requests (features, improvements, refactoring, etc.)

Reporting Issues (bugs, suggestions, documentation fixes)

Improving or adding documentation

Participating in discussions

Contribution workflow:

Fork this repository

Create a feature branch:

git checkout -b feature/your-feature-name


Commit and push changes:

git push origin feature/your-feature-name


Create a Pull Request and wait for review 🎉

We appreciate your patience and collaboration!

🧹 Code Style & Quality Rules (Code Health)

To keep the project stable and maintainable, please follow these guidelines:

Follow the project’s existing code style rules (Linters & Formatters)

Add or update unit tests when necessary (no decrease in coverage)

Code must pass CI checks before review

Use meaningful names and avoid magic numbers

Add comments for complex logic or public APIs

Typical commands before submitting:

npm run lint
npm run test


(Adjust based on actual project tooling)

📌 Pull Request Description Format (Collaboration Health)

Please structure your PR with the following template:

### 📝 What’s Changed?
- Clear summary of changes

### 🎯 Related Issue
- Reference issue number(s), e.g. #123

### 🔍 Motivation / Context
- Why is this change necessary?

### 🧪 Testing Steps
- How did you verify the change?

### 📸 Screenshots (Optional)
- For UI changes or visual updates


Review checklist:

Clear purpose and understandable diff

Changes are focused — minimal unrelated modifications

Tests included where appropriate

🐞 Issue Reporting Guidelines (Community Health)

Before submitting an Issue:

Search existing issues to avoid duplicates

Provide as much detail as possible

Bug Report Template:

### 💥 Issue Description
- What went wrong?

### 🔁 Steps to Reproduce
1. ...
2. ...
3. ...

### 📦 Environment Info
- OS:
- Node/Browser Version:
- Project Version:

### 📸 Logs / Screenshots
(Optional but very helpful)


Feature Request Template:

### ✨ What feature would you like?
- What problem does it solve?

### 🔄 Alternatives Considered
- Any other potential solutions?
