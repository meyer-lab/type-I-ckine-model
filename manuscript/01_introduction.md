## Introduction


New statistical approaches are needed to link single cell measurements to molecular pathway models. The nature of the statistical approach is dependent upon the underlying source of single cell variation. For example, stochastic differential equations effectively model processes wherein variation arises from intrinsic fluxuations (CITE). However, much of cell heterogeneity in cell response arises from variation that is extrinsic to the pathway being modeled, or even extrinsic to the cell due to variation in the extracellular environment [@doi:10.1126/science.1070919]. In this situation, the pathway of interest can be treated as deterministic, but with varying inputs. One successful approach has been to separately fit a model to each cell independently [@doi:10.15252/msb.20167137]. However, there are two limitations of this strategy: First, one usually wishes to fit both shared and varying rate parameters within a population of cells. For example, the binding affinity of a protein interaction should be shared among cells, while the level of protein expression might vary. Second, this approach requires sufficient measurements from the same cell. More often, one can easily collect many measurements in the same population of cells, but only make a limited set of measurements from an individual cell.




Other single cell fitting efforts:

- [@doi:10.15252/msb.20167137]
- [@doi:10.1016/j.celrep.2017.03.027]




Here, we present a flexible statistical approach for fitting mechanistic or data-driven models to single cell measurements. This method, which relies on moment propagation, allows for integration of single cell measurements in the same populations and of extrinsic and intrinsic variability. We apply this technique to identify how receptor variation translates to variation in response to the common gamma chain cytokine receptor cytokines. These results enable rational cytokine engineering accounting for the widespread heterogeneity within cytokine-responsive cell populations. Simultaneously, our statistical approach can be applied to many molecular programs with prominent sources of cell variability.
