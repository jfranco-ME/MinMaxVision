MAE494 Project Notes

Overall field of research:

Material discovery via atomic-scale molecular simulations.

What question are you trying to answer:

How atoms interact in different environments at the molecular level. Simulations need formulae that show how they interact with different atomic arrangements (exact grain structure math currently unknown), trying to use ML to map atomic configurations, represented as vectors, to energy and force, then find atomic separation and use it, along with statistics, to find and track the other atom’s energy states. Use quantum mechanics to obtain information needed to scale up to larger scales via E(r), where E is the energy and r is the vector.

What would change if the problem were solved quicker or more accurately:

He could generate new materials due to more solves by seeing which combinations of atoms in which configurations work for desired properties. From that, being able to determine specific materials that can be used for downstream research, as it would allow him to avoid checking every material or just choosing at random, as narrowing down the search range would save time and resources.

What equations do you solve:

He solves two equations, first is eigenvalue problems linked to electron energy, as more accurate information would be useful, second is linear equations via the activation function wx + b, since the shapes of atomic configurations can be linked by linear segments.

Is any part of the physics unknow or approximated:

He uses a matrix H with three interactions, electron-electron, electron-nuclear, and nuclear-nuclear. Electron-electron interactions are indistinguishable from each other since electrons do not have classical properties, so its dual particle-like/wave-like nature leads to approximations about how they interact that is multi-layered, which is the most fundamental part, as how they interact dictates the atomic configuration they take.

What measurement data do you have:

His measurements fall into one of two classes. They are either sparse and very accurate with little noise, as they can take many days and use hundreds of CPUs and GPUs, which is very costly in terms of both time and resources. Alternatively, they are cheap, fast, and plentiful, with tons of inaccurate data that is impossible to use for prediction, as it don’t tell anything useful due to being found using the same parameters. The ideal measurements are those that balance accuracy with efficiency.

What solver do you use:

He uses multiple solvers. PyTorch is used after data collection, and one solve would take a few days once the data from a particular run is collected. His group already trained a surrogate model to speed up the process for band-gap semiconductors, but they give up accuracy, since 0.1eV band-gap energy changes are hard to determine accurately with current hardware. His current model cannot reach that level, being able to achieve 0.2eV accuracy.

What makes the problem hard:

He said the difficulty comes from having too many electrons in a particular system and their many interactions, as the number of coefficients matters for matrix size. If you keep the matrix small, it can be solved quickly, but not enough information is obtained. Increase the matrix size, and enough information is found to be useful, but the speed of the solve is reduced significantly, which increases resource costs via memory and time.

Have you or your group tried any neural operators:

Yes, he has, and there are two levels to modeling one properly for his needs. The first level is that you can always change the loss function affected by weight parameters via regularization bounds so that it doesn’t explode, as an exploding loss function is useless. The second level is using the rudimentary features of the atomic configuration in the model, and is the most important level, as it allows him to determine which features are important by removing one and seeing if and/or how the output is affected.

What happened, where did it fail, and what do you think the cause is:

His current machine-learning model reduced the quality of the input data, but the cause of such is problem-dependent and varies based on what is being modeled. He wants high-quality data, and bad data can be helpful for training neural operators due to the wide range of qualities usually found within it, as it lets the model learn what high-quality data is by filtering said data into distinct classes based on quality.

What accuracy would actually be useful:

He said accuracy is not the ultimate goal, as it is subjective and 100% accuracy can often not be guaranteed. The threshold changes from subject to subject within an industry, different situations have different impacts due to false positives and false negatives. 1 accurate solve out of 2 billion could be useful, as accuracy cannot be put into a percentage easily since it is context-dependent, and not often helpful. He posed the question of how do you know limit of accuracy for what is it that you are working on (homework question), which is often unknowable.

What is the smallest model that would be useful:

A minimal model can be very hard to formulate since they change based on scenario, and its usefulness depends on the audience.
