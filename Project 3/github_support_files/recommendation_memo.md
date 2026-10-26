# Recommendation Memo

**To:** Dr. Houlong Zhuang  
**From:** MAE 494 Project Team  
**Subject:** Recommendation on Physics-Informed Neural Networks for Quantum Eigenvalue Problems

Dear Dr. Zhuang,

Thank you for speaking with us about your work in atomic-scale materials simulation and the computational challenges involved in obtaining accurate quantum-mechanical information for material discovery. Based on our discussion, we studied whether a physics-informed neural network (PINN) could reliably solve a simplified quantum eigenvalue problem and, more importantly, where and why the method begins to fail as the solution becomes more complex.

We used the one-dimensional time-independent Schrödinger equation in an infinite square well as a minimal test case. Although this problem does not reproduce the full complexity of many-electron materials calculations, it preserves two features relevant to your work: solving for quantum energy eigenstates and handling increasingly complex eigenfunctions. We increased the eigenstate number from low to high states as a controlled difficulty parameter.

Our conventional finite-difference baseline was extremely accurate and inexpensive. On the hardware used for this study, it computed the first ten eigenstates in approximately **4.12 ms**. The PINN accurately recovered the first three eigenstates, but reliability deteriorated beginning around the fourth state. Median wavefunction error increased to approximately **4.43% at n = 4**, **16.85% at n = 8**, and **27.09% at n = 10**. Higher states were also much more sensitive to random initialization; for example, three otherwise identical n = 8 runs produced errors ranging from approximately **0.36% to 22.4%**.

To determine whether these failures were caused by insufficient neural-network capacity, we trained the same neural-network architecture directly on the known eigenfunctions using supervised regression. The architecture was able to represent the difficult states accurately: median errors in the extended supervised capacity test were approximately **0.58% at n = 4**, **0.31% at n = 8**, and **0.32% at n = 10**. This indicates that the network itself had sufficient representational capacity and that the dominant limitation in our experiment was the **physics-informed optimization process**, rather than an inability of the architecture to represent oscillatory solutions. Error-spectrum analysis supported this conclusion: failed PINN solutions contained substantial distortions in neighboring and higher eigenmodes rather than simply reverting to lower-frequency states.

For this specific forward eigenvalue problem, we therefore **would not recommend replacing a conventional eigensolver with a standard PINN**. PINN training required a median of approximately **19.3 s per eigenstate**, thousands of times longer than the conventional solution, while also becoming less reliable for higher states. We have high confidence in this conclusion for the simplified problem because it was supported by an analytical solution, a converged numerical baseline, multiple random seeds, and a controlled expressivity-versus-optimization experiment. Our confidence is lower when extrapolating directly to realistic materials simulations because the model omits many-electron interactions, realistic potentials, three-dimensional structure, and the large parameter spaces present in your research.

We do, however, think further investigation of **neural operators or parameterized surrogate models** could be worthwhile if the goal is to accelerate a large family of expensive simulations rather than replace a solver for one fixed eigenproblem. A useful next experiment would be to define a family of one-dimensional Schrödinger problems with varying potential fields or material parameters and compare a neural operator or other surrogate against the conventional solver across many queries. That experiment would more closely reflect the many-query material-screening setting you described and would allow a meaningful comparison of training cost, accuracy, and break-even number of evaluations.

Thank you again for your time and for helping us connect the computational experiment to a real materials-discovery problem.

Sincerely,  
**MAE 494 Project Team**
