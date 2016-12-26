General BagOfWords classification solution - functional for arbitrary dataset (any given dataset).
Dataset has to be sorted in folders by class names.

To run feature extraction pass dataset path as argument:
	- FeatureExtraction.py -t DATA/train-400

To run classification pass dataset along with BoW.pkl file (optional visualization):
	- Classification.py -b BoW-train-400.pkl -t DATA/test-200
	- Classification.py -b BoW-train-400.pkl -t DATA/test-200 -v
