to run the script, edit the config_1.json file.
the actual command I run is python main.py --config config_1.json --cmdln 1 however it may differ for you
{
	"size": [16,6,2],
	"learning_rate": 0.0001,
	"epochs": 1000,
	"data_file": ["L30fft16.out"],
	"activation": "tanh",
	"momentum": "1",
	"softmax_last": "0"
}
the size can be an array or any size
the learning rate can be any real number
the epochs can also be any real number
the acitvation defaults to softmax if "tanh" is not entered
for both momentum and softmax 1 means it will be used and 0 means it will not