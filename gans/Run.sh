## - DO NOT RUN THIS YET ! 

git clone https://github.com/davidADSP/GDL_code.git

conda create -n generative python=3.6 ipykernel
conda activate generative

#the generative env is created here: /home/anupam/miniconda3py38/envs 

pip install -r GDL_code/requirements.txt 

# ----- 
#got this error: ERROR: Could not find a version that satisfies the requirement tensorflow==1.14.0 from <etc>
#did: pip install tensorflow

# Downloading tensorflow-2.11.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (588.3 MB)  (and got no space left on device!)
# cleaned up my downloads .tar.xzs, chromium src (27G!) etc.

# works: python : import keras (calls into tensorflow)
# note: jupyter notebook wasn't working (notebook sub_cmd wasn't available so I pip install notebook and reran cmd)
# -----




