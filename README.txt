This file contains instructions on how to run this code form scratch.

Prerequizits:
- Anaconda: this is the may backbone for python training and allows other uses to share their virtual enviroment easily
- VScode: Any IDE can be used since most of the commands and edits will be executable through the conda terminal. 
          However, conda has Vscode support so for simplicity I recomend this IDE.
- VLC: This will read the WAV file generated form the code. 

Firstly, ensure that you have a yml file named requirements.yml. 
This holds the conda environment and avoids the pit falls of out dated code.

Then run the following command to create the conda virtual environment:

    # Create environment from the YML file
    conda env create -f requirements.yml

    # Activate the environment
    conda activate music_omr_gpu

    # Install any additional pip packages
    pip install -r requirements.txt

The original code for the classification python code used is in the zip file "MusicSymbolClassifier-master" but 
the content has been edited to match with the lybrary versions in the virtual environment created.


Then you are ready to use the code and its environment fuctions.
Do note that I am using a tensorflow-gpu version that is compatible with my gtx 1080, newer cards might cause issues.

To dramatically shorten the length of time to train a model, to read and play simple sheet music,
first ensure that you are in the Simple_model_trainer folder, then copy and paste this:
	
python TrainModel.py `
>>     --dataset_directory data `
>>     --model_name vgg4 `
>>     --width 96 `
>>     --height 192 `
>>     --minibatch_size 32 `
>>     --optimizer Adadelta `
>>     --use_fixed_canvas `
>>     --use_existing_dataset_directory

These set of comands gives the code instructions to use the edited information present in the file.

If you want to run the full model, although its accuracy is low, you can run the code below. However, this command re-writes the content of data file, and the model created is not compatible with the current sound generating code. 
     
	python TrainModel.py --model_name vgg4 --minibatch_size 32

But for simplicity run the previous version, it should take roughly 2 hours. 


A file with the extension .h5 should appear in the same file as the TrainModel.py code, that is the model.
You can rename the file to what ever you like but the one I used is named Final_vgg4.h5


Next is the sound generation.
Ensure that you in the parent file and you should see the code music_pipeline.py. 
Enter that file location in conda and run this comand:

    python music_pipeline.py --sheet input_sheet.png --classifier Final_vgg4.h5

This code does two things, it runs the note_scanner code which uses the model to scan and make a report on the input_sheet.png
Then it sends that report to the midi_creation.py code which converts the report into sound. 
A file named Generated_Observations will contain all the notes found and classified from the sheet music, including the final report.
If you run this code again, you will see the same file, but numbered differently so that different sheet musics can scanned and stored with confusion.

