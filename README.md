# Deep Spatiao Temporal Generative Network </p><br>
*Donggeun Park<sup>a</sup>, Jaemin Lee<sup>a</sup>, Hugon Lee<sup>a</sup>, Grace X. Gu<sup>b</sup>, and Seunghwa Ryu<sup>*a</sup>**
a. Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Daejeon 34141, Republic of Korea
<br>
b. Department of Mechanical Engineering, University of California, Berkeley, CA 94720, USA

![coverfigure](https://github.com/DonggeunPark/DG/assets/131414228/b8b30fe0-185f-45bb-bc21-7933fa3a41fe)

# STGNet
The majority of AI-based surrogate models are limited to predicting either the overall S-S curve or local stress/strain fields at certain strain levels, offering only a fragmentary view of the entire mechanical failure process. Understanding the complete spatiotemporal dynamics, encompassing crack propagation and local stress/strain evolution, is pivotal not merely for enhancing the performance of CMs, but also for early detection of local fracture sites. We introduce the deep-Spatial and Temporal Generative Network model (STGNet). STGNet is a DL model proficient in predicting spatiotemporal dynamics, such as stress evolution and crack propagation, from the morphological configuration features of a CM. Here, we select binary composites for their ability to represent diverse morphological patterns and train STGNet using crack phase fields simulations on randomly patterned CMs. The trained STGNet is tested on unseen configurations, exhibiting the extrapolated mechanical properties within the design space far beyond a realm explored in the training data. The extrapolation task is considered a critical issue for CM optimization research. STGNet significantly accelerates the inference time by a factor of 60,000, while achieving accuracy comparable to FEM. Compared to conventional autoregressive DL model as a baseline, it accurately predicts the physically critical locations over entire time steps for various fracture mechanisms, such as reinforcement breakage and matrix cracking, contributing to the CMs with exceptional mechanical properties. 

# Results
![z2](https://github.com/DonggeunPark/DG/assets/131414228/cfddcd6b-aaab-4c83-8d75-3c0f5fa874b7)

![z3](https://github.com/DonggeunPark/DG/assets/131414228/ea49bfb8-f7e7-4de9-aa5c-a078682d67ac)

