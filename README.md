![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)

# pyJedAI: A Library with Resolution-related Structures and Procedures for Products

This project is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [Apache 2.0 License](LICENSE).

The software and data in this repository are associated with the paper [pyJedAI: A Library with Resolution-related Structures and Procedures for Products](https://doi.org/10.1287/ijoc.2023.0410) by Ekaterini Ioannou, Konstantinos Nikoletos and George Papadakis. 

## Version

The version used in the paper is

[![Release](https://img.shields.io/github/v/release/INFORMSJoC/Template?sort=semver)](https://github.com/AI-team-UoA/pyJedAI/releases/tag/0.1.7)

## Cite

To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2023.0410) and the software, using the following DOI.

<!--
[![DOI](https://zenodo.org/badge/285853815.svg)](https://zenodo.org/badge/latestdoi/285853815)
--->

```
@misc{pyjedaiProductMatching,
  author =        {Ekaterini Ioannou, Konstantinos Nikoletos, George Papadakis},
  publisher =     {INFORMS Journal on Computing},
  title =         {pyJedAI: A Library with Resolution-related Structures and Procedures for Products},
  year =          {2024},
  doi =           {10.1287/ijoc.2023.0410.cd},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0410},
}
```

## Authors

- [Ekaterini Ioannou](https://www.tilburguniversity.edu/staff/ekaterini-ioannou), Assistant Professor at Tilburg University, The Netherlands 
- [Konstantinos Nikoletos](https://nikoletos-k.github.io), Research Associate at University of Athens, Greece
- [George Papadakis](https://gpapadis.wordpress.com), Senior Researcher at University of Athens, Greece

## Description

This work presents an open-source Python library, named pyJedAI, which provides functionalities supporting the creation of algorithms related to Product Entity Resolution. Building over existing state-of-the-art resolution algorithms (Papadakis et al. 2021a), the tool offers a plethora of important tasks required for processing product data collections. It be can easily used by researchers and practitioners for creating algorithms analyzing products, such as real-time ads bidding, sponsored search, or pricing determination. In essence, it allows to easily import product data from the possible sources, compare products in order to detect either similar or identical products, generate a graph representation using the products and desired relationships, and either visualize or export the outcome in various forms. Our experimental evaluation on data from well-known online retailers illustrates high accuracy and low execution time for the supported tasks. To the best of our knowledge this is the first Python package to
focus on product entities and provide this range of Product Entity Resolution functionalities. 

## Building

In Linux, to build the version used for this paper, execute the following two steps.

1. Create and then activate a conda environment with Python 3.9 or 3.10:
```
conda create --name pyJedAI_env python==3.9
conda activate pyJedAI_env
```

2. Then, install the tool using either:
```
pip install pyjedai==0.1.7
```

or in the root directory using:
```
git clone https://github.com/AI-team-UoA/pyJedAI.git
pip install . 
```

Please note that it requires py_stringmatching, which can be installed (before step 2)
using command:
```
conda install conda-forge::py_stringmatching
```

## Usage

As describe in the journal, the tool implements a comprehensive end-to-end process for realizing possible similarity relation operators. The process, shown the above figure, consists of four
steps: 
1. data reading,
2. filtering,
3. verification, and
4. data writing and evaluation.

<div align="center">
  <img width="771" align="center" alt="pLibTool" src="https://github.com/user-attachments/assets/62ae3ff2-79f9-49e2-812c-4baa37571cf9">
</div>


### __Google Colab Hands-on demo:__ 

The simplest way to reproduce and view the results of this paper, is using the Colab notebook here:     <a href="https://colab.research.google.com/drive/1VB_DfIT3eLXhlg6vGZSCWrJKc7AcLIpA?usp=sharing">
        <img align="center" src="https://3.bp.blogspot.com/-apoBeWFycKQ/XhKB8fEprwI/AAAAAAAACM4/Sl76yzNSNYwlShIBrheDAum8L9qRtWNdgCLcBGAsYHQ/s1600/colab.png" width=80/> 
    </a>


<div align="center">

</div>

Alternatively first run the installation and then go to `src` directory and run:

- **Blocking-based workflow**: `python blocking_workflow.py --dataset 'Abt - Buy'`
- **Similarity join-based workflow**: `python similarity_joins_workflow.py --dataset 'Amazon - Google Products'`
- **Nearest neighbor-based workflow**: `python nn_workflow.py --dataset 'Abt - Buy' --schema 'schema-agnostic' `

where for
- `--dataset` flag, available values are `{'Abt - Buy', 'Amazon - Google Products', 'Wallmart - Amazon' }` and for
- `--schema` flag `{'schema-agnostic', 'schema-based'}`, available only for the NN workflow.

For the scalability test:

```
python dbpedia_scalability.py
```

## Ongoing Development

This main tool is being developed on an on-going basis at the author's
[Github site](https://github.com/AI-team-UoA/pyJedAI).

## Documentation page

To view more examples of this software visit [readthedocs](https://pyjedai.readthedocs.io/en/latest/intro.html) website. 

## Support

For support in using this software, submit an
[issue](https://github.com/AI-team-UoA/pyJedAI/issues/new).
