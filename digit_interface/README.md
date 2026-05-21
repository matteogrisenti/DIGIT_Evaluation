# DIGIT_Windows_Interface

This repository implement the [[digit-interface](https://github.com/facebookresearch/digit-interface)] official repository of Meta-AI but for Windows system. The original one work for Linux system.
The GUI is inspired by the github repository [[DIGIT-GUI](https://github.com/gemixin/digit-gui)] by *gemixin*. 

## Installation
1: Clone the repository.
1.5: Create a virtual environment and activate it.
```bash
python -m venv myenv
source myenv/Scripts/activate  
```
2: Install the required dependencies using
```bash 
pip install -r requirements.txt
```
3: Run the GUI using
```bash
python gui.py
```

## Citation
**DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation**  
Mike Lambeta, Po-Wei Chou, Stephen Tian, Brian Yang, Benjamin Maloon, Victoria Rose Most, Dave Stroud, Raymond Santos, Ahmad Byagowi, Gregg Kammerer, Dinesh Jayaraman, Roberto Calandra  
_IEEE Robotics and Automation Letters (RA-L), vol. 5, no. 3, pp. 3838–3845, 2020_  
[https://doi.org/10.1109/LRA.2020.2977257](https://doi.org/10.1109/LRA.2020.2977257)