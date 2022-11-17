# Story-Generation-Using-GPT2

If you want to retrain the model, the data are present here at https://drive.google.com/drive/folders/1v_UdPlmJzFM7GWVb988iIYIDEwv0kDuT?usp=sharing. The data need to be loaded in `kaggle/working/` directory.

This requires dependency and you can install that using `pip install requirements.txt` and the `pytorch_model.bin`model can be downloaded from the same above link and cloning the repo for other tokenizer.  

To directly use the model for inferencing, run the code
`streamlit run storyGeneration.py`

It provide an interface using, to type the prompt and it will return the story regarding this. 