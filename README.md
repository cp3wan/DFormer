# DFormer: Diffusion-guided Transformer for Universal Image Segmentation

Hefeng Wang, Jiale Cao, Rao Muhammad Anwer, Jin Xie,
Fahad Shahbaz Khan, Yanwei Pang
<td align="center"><a href="https://arxiv.org/abs/2306.03437">	arXiv:2306.03437 </a></td>
</tr>
</tbody></table>


## Abstract
This paper introduces an approach, named DFormer, for universal image segmentation. The proposed DFormer views universal image segmentation task as a denoising process using a diffusion model. DFormer first adds various levels of Gaussian noise to ground-truth masks, and then learns a model to predict denoising masks from corrupted masks. Specifically, we take deep pixel-level features along with the noisy masks as inputs to generate mask features and attention masks, employing diffusion-based decoder to perform mask prediction gradually. At inference, our DFormer directly predicts the masks and corresponding categories from a set of randomly-generated masks. Extensive experiments reveal the merits of our proposed contributions on different image segmentation tasks: panoptic segmentation, instance segmentation, and semantic segmentation. Our DFormer outperforms the recent diffusion-based panoptic segmentation method Pix2Seq-D with a gain of 3.6% on MS COCO val2017 set. Further, DFormer achieves promising semantic segmentation performance outperforming the recent diffusion-based method by 2.2% on ADE20K val set.
![](show.jpg)

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for DFormer](datasets/README.md).

See [Getting Started with DFormer](GETTING_STARTED.md).



## Model Zoo and Baselines

We provide the baseline results and trained models available for download.
## COCO Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: dformer_R50_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/dformer_R50_bs16_50ep.yaml">DFormer</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">51.1</td>
<td align="center"><a href="https://pan.baidu.com/s/1-nS9BVvemRz20oB8iABhkA?pwd=xg6r">model</a></td>
</tr>
<!-- ROW: dformer_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/panoptic-segmentation/swin/dformer_swin_tiny_bs16_50ep.yaml">DFormer</a></td>
<td align="center">Swin-T</td>
<td align="center">50</td>
<td align="center">52.5</td>

<td align="center"><a href="https://pan.baidu.com/s/1em8yVsaFbQjvGSJ5qVT88w?pwd=8gfq">model</a></td>
</tr>
</tbody></table>


### Instance Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: dformer_R50_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/dformer_R50_bs16_50ep.yaml">DFormer</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">42.6</td>
<td align="center"><a href="https://pan.baidu.com/s/1arjRIxfqpnjqaYOG0W6r9g?pwd=9pah">model</a></td>
</tr>
<!-- ROW: dformer_swin_tiny_bs16_50ep -->
 <tr><td align="left"><a href="configs/coco/instance-segmentation/swin/dformer_swin_tiny_bs16_50ep.yaml">DFormer</a></td>
<td align="center">Swin-T</td>
<td align="center">50</td>
<td align="center">44.4</td>
<td align="center"><a href="https://pan.baidu.com/s/1YcOdvacuWbOIewmByybN2Q?pwd=ewgk">model</a></td>
</tr>
</tbody></table>


## ADE20K Model Zoo


### Semantic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: dformer_R50_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/dformer_R50_bs16_160k.yaml">DFormer</a></td>
<td align="center">R50</td>
<td align="center">160k</td>
<td align="center">46.7</td>
<td align="center"><a href="https://pan.baidu.com/s/14I9sU9jDpn8tq557ucfEaw?pwd=j5iq">model</a></td>
</tr>
<!-- ROW: dformer_swin_tiny_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/dformer_swin_tiny_bs16_160k.yaml">DFormer</a></td>
<td align="center">Swin-T</td>
<td align="center">160k</td>
<td align="center">48.3</td>
<td align="center"><a href="https://pan.baidu.com/s/1fBUnQ0gMfeJmxhRjRegPBg?pwd=4cfr">model</a></td>
</tr>

</tbody></table>



## <a name="CitingMask2Former"></a>Citing DFormer

If you use DFormer in your research or wish to refer to the baseline results published in the Model Zoo and Baselines, please use the following BibTeX entry.

```BibTeX
@article{wang2023dformer,
  title={DFormer: Diffusion-guided Transformer for Universal Image Segmentation},
  author={Wang, Hefeng and Cao, Jiale and Anwer, Rao Muhammad and Xie, Jin and Khan, Fahad Shahbaz and Pang, Yanwei},
  journal={arXiv preprint arXiv:2306.03437},
  year={2023}
}
```


## Acknowledgement
Many thanks to the nice work of Mask2Former @[Bowen Cheng](https://bowenc0221.github.io/) and DDIM @[Jiaming Song](http://tsong.me). Our codes and configs follow [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [DDIM](https://github.com/ermongroup/ddim).
