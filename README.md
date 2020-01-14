# stmod
Streamer modelling software. At early stage of development.

Finite elements computations based in [Deal.II](https://github.com/dealii/dealii) framework.


## Now done
Mesh generation over gmsh. Realistic geometry.
Poisson equation solver on axially symmetric geometry.


## Active tasks
### Hi priority

[x] Solution-like fields sampling with gradients and laplacians

[x] Vector field output class from FESampler

[x] Solution-like vector interpolation for refine <- Use SolutionTransfer class

[x] Scalar field output class

[ ] Electrons emission possibility

[ ] Add right hand side in cyllindric geometry for poisson

[ ] Fix laplacian for cylliongric geomentry

[ ] Add output hook

### Low priority

[x] Manifold ids for grid near the needle

[ ] Manifold ids for boundary -- ask Fedor

[x] Laplacian output test
