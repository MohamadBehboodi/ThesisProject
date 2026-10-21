# Optimizing the DLSTM profit prediction model using optimization algorithms.
# Athor: Mohamad Hasan Behboodi
# S.N: 40212441279
# email: mohamadbehboodi.isf@gmail.com
# Allameh Tabatabee University of Tehran

# Intro
This is the completed and reviewd version of models architecture with loading all data. By the following steps, you can run this project. First you need to  provide project requirements: 

**
    Extract folder contents compeletely and then use a linux server or a server which has WSL to run project on it. This point is important because files have been locked with "fcntl" function. If you don't have access to a linux server or WSL, you'll need to change this part of the code.
**

After extracting folder contents in the ZIP file, You can install all of project requirements by the following command (Run it on the terminal or command prompt):

    python -m pip install -r requirements.txt
    python run.py


If you want to use "Jupyter Lab" for running notebooks, open each notebook files which have been located on the "notebooks" folder in order (Start from file which its name starts with 00 and the last file will be file which its name starts with 10). Running "07 file" is optional. All notebooks configs are setted on "configs/default.json" and results will be saved at "outputs/main" directory.

** Attention!
    Before running main project, install a CUDA-compatible torch build (contains PyTorch) on the server. Trial version could be ran on CPU but the main version needs a GPU server.
    Also if the connection lost for any reason, the model execution process must not stop. To insure this, run project commands on the "tmux" environment.
** 

# Project Settings
- Layers Width: [4000, 2000, 2000, 30]
- Epoch: 30
- Population: 200
- Other optimizers budget:  50000 evaluation
- TAR_CHOA budget: 300000 evaluation
- Seeds: [17, 29, 43]
- Validation years: [1396, 1397, 1398, 1399, 1400, 1401]
- Device: cuda

# Scientific Points and Backup Notes

This project is an independent re-implementation of original model and the original code which has been written by the base article athors is not accessible. For designing models, formulas and definitions in the base article have been used. In addition to this matter, the original codes for ARGA/ARWOA was not accessible and instead of them GA and WOA have been used.

All changes has been explained at the "docs/Method_Decisions.md".   

Last cell in each notebook is specified for restoring results based on cells outputs of the notebook. But weights will not be restored by this cell. "Checkpoints" and "model.pt" must be saved separately. Last cell will save only important data which is in notebook outputs.

In the "docs/previous_release_validation" directory results for previous version of the project has been saved and it doesn't have new train/test process.   

Run `python run.py preflight` before final running to detect that server resources support model process.
