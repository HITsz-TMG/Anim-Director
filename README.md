<div align="center">

<h2>Anim-Director: A Large Multimodal Model Powered Agent for Controllable Animation Video Generation </h2> 

 <b> SIGGRAPH Asia 2024 </b>
<!-- ## <b><font color="red"> TaleCrafter </font>: Interactive Story Visualization with Multiple Characters</b> -->


_**[YunXIn Li](https://github.com/YunxinLi), [Haoyuan Shi](https://github.com/HaoyuanShi), Baotian Hu, Longyue Wang,<br>Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang***_
  
(* Corresponding Authors)



<p align="center"> <img src="assets/demo1.gif" width="700px"> </p>
 
</div>


## 🎏 Abstract
<b>TL; DR: <font color="red">Anim-Director</font> is an autonomous animation-making agent where LMM interacts seamlessly with generative tools to create detailed animated videos from simple narratives.</b>

> Traditional animation generation methods depend on generative tools and human-labelled data, requiring a sophisticated multi-stage pipeline that demands substantial human effort and incurs high training costs. These methods typically produce brief, information-poor, and context-incoherent animations due to limited prompting plans. To overcome these limitations and automate the animation process, we introduce large multimodal models (LMMs) as the core processor to build an autonomous animation-making agent, named Anim-Director. This agent harnesses the advanced understanding and reasoning capabilities of LMMs and external tools to create detailed animated videos from concise narratives or simple instructions. Specifically, it operates in three main stages: Firstly, the Anim-Director generates a coherent storyline from user inputs, followed by a detailed director’s script that encompasses character profiles, interior/exterior descriptions, and context-coherent scene descriptions including appearing characters, interiors or exteriors, and scene events. Secondly, we employ LMMs with the image generation tool to produce visual images of settings and scenes. These images are designed to maintain visual consistency across different scenes using a visual-language prompting method that combines scene descriptions and images of the appearing character and setting. Thirdly, scene images serve as the foundation for producing animated videos, with LMMs generating prompts to guide this process. The whole process is notably autonomous without manual intervention, as the LMMs interact seamlessly with generative tools to generate prompts, evaluate visual quality, and select the best one to optimize the final output. To assess the effectiveness of our framework, we collect varied short narratives and incorporate various image/video evaluation metrics including visual consistency and video quality. The experimental results and case studies demonstrate the Anim-Director’s versatility and significant potential to streamline animation creation.

  
## ⚔️ Overview

<p align="center"> <img src="assets/overview.png" width="700px"> </p>
Given a narrative, Anim-Director first polishes the narrative and generates the director’s scripts using GPT-4. GPT-4 interacts with the image generation tools to produce the scene images through Image + Text → Image. Subsequently, the Anim-Director produces videos based on the generated scene images and textual prompts, i.e., Image + Text → Video. To improve the quality of images and videos, we realize deep interaction between LMMs and generative tools, enabling GPT-4 to refine, evaluate, and select the best candidate by self-reflection reasoning pathway.


## 📀 Visual Example

<p align="center"> <img src="assets/visualeg.png" width="700px"> </p>
A visual example of Anim-Director.


## 🌰 More Examples

<div align="center">
<video src="https://github.com/HITsz-TMG/Anim-Director/blob/9a7e7bd6e4ada44eaeabeed6f3bd173bf2a1dd19/assets/demo1.mp4" controls="controls" width="500" height="300"></video>
</div>
<div align="center">
<video src="https://github.com/HITsz-TMG/Anim-Director/blob/9a7e7bd6e4ada44eaeabeed6f3bd173bf2a1dd19/assets/demo2.mp4" controls="controls" width="500" height="300"></video>
</div>
<div align="center">
<video src="https://github.com/HITsz-TMG/Anim-Director/blob/9a7e7bd6e4ada44eaeabeed6f3bd173bf2a1dd19/assets/demo3.mp4" controls="controls" width="500" height="300"></video>
</div>


## Citation
```bib
@misc{li2024animdirector,
      title={TaleCrafter: Interactive Story Visualization with Multiple Characters}, 
      author={YunXIn Li and Haoyuan Shi and Baotian Hu and Longyue Wang and Jiashun Zhu and Jinyi Xu and Zhen Zhao and Min Zhang},
      year={2024},
}
```
