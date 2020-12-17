# A pipeline to identify homogeneous subgroups of patients and important features using topological data analysis integrated with machine learning

Ewan Carr<sup>1</sup>  
Mathieu Carrière<sup>2</sup>  
Bertrand Michel<sup>3</sup>  
Fred Chazal<sup>4</sup>  
Raquel Iniesta<sup>1</sup>  

<sup>1</sup> Department of Biostatistics and Health Informatics,
Institute of Psychiatry, Psychology \& Neuroscience, King's College London, United Kingdom.  
<sup>2</sup>Inria Sophia-Antipolis, DataShape team, Biot, France.  
<sup>3</sup>Ecole Centrale de Nantes, LMJL -- UMR CNRS 6629, Nantes, France.  
<sup>4</sup>Inria Saclay, Ile-de-France, France.  

Please contact <raquel.iniesta@kcl.ac.uk> for queries.

### Abstract

> **Motivation:**  
> There is considerable interest in identifying homogeneous patient subgroups
with common clinical and biological characteristics or outcomes. Better
detection of at-risk groups may allow earlier intervention and more targeted
treatments. In this paper we exploit recent developments in topological data
analysis to present a pipeline for clustering based on the Mapper algorithm.
Topological data analysis is a growing field that offers tools to infer,
analyse, and exploit the shape of data. The Mapper algorithm is one promising
application that seeks to identify shapes (i.e. clusters) in a multidimensional
cloud of data points, herein referred to as topological features. Our pipeline
aims to identify subgroups sharing similar characteristics of interest based on
their membership to topological features. A central benefit of our approach is
the ability to incorporate prior knowledge or clinical and biological data to
inform the clustering process and the selection of optimal clusters. By
specifying a *filter* (a lens through which to view the data) and selecting
a target outcome on which to rank and evaluate candidate clusters users can
identify meaningful clusters of clinical relevance. Other advantages of this
pipeline include the use of the bootstrap to restrict the search to robust
topological features; the use of machine learning to inspect clusters; and the
ability to incorporate mixed data types.   
>
> **Results:** We present a pipeline to identify and summarise
statistically significant topological features from a point cloud using
Mapper. The Mapper graph and selection of optimal clusters are informed
by a user-specified predictor or outcome of interest.\
