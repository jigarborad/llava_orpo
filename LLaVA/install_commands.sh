pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install datasets
pip install autoawq@https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.0/autoawq-0.2.0+cu118-cp310-cp310-linux_x86_64.whl