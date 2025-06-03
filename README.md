## Patameter-Aware Contrastive Knowledge Editing [ACL 2025]

### How to run
1. Download the original LLMs and store them locally: GPT-J, Llama2 (7B), Llama3 (8B)
2. Update the corresponding model path in `CONFIG`
3. Perform editing by executing `python pac.py --llm_name gpt-j --data_name zsre`, (You can specify the `llm_name` and `data_name` according to your requirements.)

### Citation
```
@inproceedings{zhaimengqi,
    author = {Songlin Zhai and Yuan Meng and Yuxin Zhang and Guilin Qi},
    title = {Parameter-Aware Contrastive Knowledge Editing: Tracing and Rectifying based on Critical Transmission Paths},
    booktitle = {ACL},
    year = {2025}
}
```
