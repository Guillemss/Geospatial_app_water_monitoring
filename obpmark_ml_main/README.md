# OBPMark-ML

**Machine Learning Benchmarks for On-Board Space Processing**

OBPMark-ML is an open-source benchmark suite for evaluating ML inference performance
on space and embedded hardware. It is developed as part of [OBPMark](http://obpmark.org),
the On-Board Processing Benchmarks initiative by ESA.

📖 **Full documentation: [obpmark.gitlab-pages.bsc.es/obpmark-ml](https://obpmark.gitlab-pages.bsc.es/obpmark-ml/)**

## Tasks

| Task | Model | Dataset |
|------|-------|---------|
| Ship Detection | YOLOX-Tiny | Airbus Ship Detection |
| Cloud Segmentation | UNet (width-scaled) | Cloud-95 |

## Quick Start

```bash
git clone https://bsc.gitlab.es/obpmark/obpmark-ml.git
cd obpmark-ml
pip install -e .[onnx]        # or: tflite, tensorflow, pytorch, all
./get_data.sh                 # downloads both datasets
python benchmark.py --task object_detection --framework onnx --precision fp32
```

## Cite

If you use OBPMark-ML in your work, please cite:

```bibtex
@inproceedings{wolf2025obpmarkml,
  author    = {Wolf, Jannis and Sol\'{e}, Marc and Rodriguez, Ivan and Kosmidis, Leonidas and Steenari, David},
  title     = {Design and Implementation of an Open Source Machine Learning Benchmarking Suite for On-board Space Systems},
  booktitle = {Proceedings of the 2025 European Data Handling \& Data Processing Conference (EDHPC)},
  year      = {2025},
  pages     = {1--7},
  note      = {IEEE Paper ID: 11326405. DOI pending.},
}
```

## Contact

[jannis.wolf@bsc.es](mailto:jannis.wolf@bsc.es)

## Contributors

- Jannis Wolf — Barcelona Supercomputing Center (BSC)
- Leonidas Kosmidis — BSC / UPC
- David Steenari — ESA
