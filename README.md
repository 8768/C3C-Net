This is a Pytorch implementation of C3C-Net. Now the corresponding paper is available online at https://ieeexplore.ieee.org/document/9338307.

# Introduction
Understanding the distribution of crowd flow is crucial for optimizing intelligent transportation services such as ride-hailing and sharing-bike scheduling. A key element for these applications is the accurate and reliable prediction of origin-destination (OD) flows, which provide a detailed profile of how and when passengers enter and exit the service. Despite the interest in traffic prediction in recent years, OD flow estimation remains a challenging task for deep spatiotemporal networks due to the two unique characteristics: (i) City-wide OD human movements are quite sparse. OD flows between many region pairs could be zero over a long period, the data sparsity and representation is a problem for deep models, (ii) OD spatial relations are timesensitive and large-scale, and can affect region-level similarity, attractiveness, and pair-level correlation. To solve these issues, we propose a context-aware 3D criss-cross fusion network for city-wide OD crowd flow prediction (C3CNet). First, we design a novel two-stream representation framework, where a contextual-embedded pair stream and a geographical-based region stream maintain both OD and in/outflow distribution from generation/attraction and transaction perspectives while preserving dynamic contextual factors. The information involved for OD demands as data augmentation can help to solve the sparsity problem adaptively in the learning process. Then, the C3C-Net consists of several components during the learning process, such as 3D  module and residual mechanism. In particular, a 3D criss-cross attention module to learn global multi-view spatiotemporal dependencies and heterogeneity simultaneously with high computing efficiency. We further propose a stream fusion attention module to enhance the model performance by combining multi-scale region and pair-level latent features.

# Contribution Guide

Thank you for considering contributing to this project.
We welcome all contributions including code, documentation, and community feedback.

This guide explains:
1.How external developers can contribute 
2.Code rules and style guidelines
3.Pull Request submission format 
4.Issue reporting best practices  


---

## 1. How to Contribute

We follow the standard GitHub workflow for contributions:

1. **Fork** the repository  
2. **Clone** your fork to local machine:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
3. **Create a new branch** for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Describe the change briefly"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a **Pull Request** on GitHub

We will review your PR and provide feedback as soon as possible.

---

## 2. Code Style & Quality Guidelines

To keep the project consistent and maintainable:

1.Follow the coding style already used in this repository

2.Ensure changes pass **Lint checks & Unit tests**

3.Avoid unnecessary files or unrelated changes in the PR

4.Provide comments for complex logic

5.Commit messages should be clear and meaningful

Recommended commands before submitting:
```bash
npm run lint
npm run test
```


---

## 3. Pull Request Description Format

To ensure efficient review, please fill out your PR like this:

```
###What’s Changed?
A clear and concise description of the modifications.

###Related Issues
Link to issue(s): #issue-number

###Motivation / Context
Why is this important?

###Testing Steps
How to reproduce and validate the change:

1. Step one
2. Step two
3. ...

###Screenshots (Optional)
Attach screenshots for UI-related changes.
```

PR Checklist:

1.Code works as expected  
2.Tests updated (if needed)  
3.No style/lint issues  
4.PR title and commit messages are meaningful  

---

## 4. Issue Reporting Guidelines

Before creating an Issue:

1.Check existing issues
2.Provide a clear description and reproduction
3.Include environment info if applicable

Bug Report Template:

```
###Issue Description
What happened?

###Steps to Reproduce
1. ...
2. ...

###Environment Info
- OS:
- Browser/Node Version:
- Project Version:

###Logs / Screenshots
(Optional)
```

Feature Request Template:

```
###Feature Description
What would you like to add or improve?

###Why is this needed?
Explain the value of the feature.

###Alternatives
Any alternative solution considered?
```

---

## Thanks for Contributing

Every contribution helps improve the project. Thank you for your support! 
If you have any questions, feel free to open an Issue or discussion.
