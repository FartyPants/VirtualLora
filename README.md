# VirtualLora

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q5MOB4M)

extension for text WebUI

It has to be in extensions/VirtualLora

(it has hardcoded folders so don't change VirtualLora)

News: Added Strength for Lora (must ve set before hitting Load lora)
![image](https://github.com/user-attachments/assets/5de205bb-890c-4299-919f-6332bf068de8)


![image](https://github.com/FartyPants/VirtualLora/assets/23346289/2aa0d0d6-7288-4179-99e7-d8e60c8187be)

This is to create your own collection folders to sort hundreds of Loras the way you want

# Custom Sets

You can create the collection tree by simpply writing a template txt file (collection set) in Setup tab. You can have many of these collection sets and each will have an root items (like a virtual folder) where you can type anything, then the actual LORA folders.

example of collection set:

```
CodeLlama
+ L2-Sydney_UASST_CodeLlama_v2
+ L2-Sydney_UASST_CodeLlama_BAD

Llama 2
+ Sydney_V4_LLaMA2_Base_S_UASST_512  #Some comment about the LORA

Mix
+ L1-CAS_radio_Sydney_128_v4
```
This will display three folders in Root (Collections) column:

- CodeLlama
- Llama 2
- Mix

![image](https://github.com/FartyPants/VirtualLora/assets/23346289/c952ab9e-0113-4213-99a5-5c8bee4e9543)

You can add comments to the each lora with # 

+ Sydney_V4_LLaMA2_Base_S_UASST_512  #Needs some more testing



