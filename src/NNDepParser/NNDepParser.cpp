#include <iostream>
#include <chrono>
#include <omp.h>

#include "Argument_helper.h"
#include "NNDepParser.h"

using namespace std;

DepParser::DepParser() {
	srand(0);
}

DepParser::~DepParser() {
}

void DepParser::createAlphabet(const vector<Instance>& vecInsts) {
	cout << "Creating Alphabet..." << endl;
	int maxsize = vecInsts.size();
	unordered_map<string, int> word_stat;
	unordered_map<string, int> depLabel_stat;
	for (int idx = 0; idx < maxsize; idx++) {
		const Instance& inst = vecInsts[idx];
		int wordSize = inst.words.size();
		for (int idy = 0; idy < wordSize; idy++) {
			const string& word = inst.words[idy];
			word_stat[word]++;
		}
		word_stat[unknownkey] = m_options.wordCutOff + 1;

		int depLabelSize = inst.result.labels.size();
		for (int idy = 0; idy < depLabelSize; idy++) {
			const string& label = inst.result.labels[idy];
			depLabel_stat[label]++;
		}
	}
	m_driver._hyperparams.labelAlpha.initial(depLabel_stat, 0);
	m_driver._hyperparams.wordAlpha.initial(word_stat, m_options.wordCutOff);
	m_driver._hyperparams.extWordAlpha.initial(m_options.wordEmbFile);

	m_driver._hyperparams.extWordAlpha.set_fixed_flag(true);
	m_driver._hyperparams.wordAlpha.set_fixed_flag(true);
	m_driver._hyperparams.labelAlpha.set_fixed_flag(true);

	int tmpID1 = m_driver._hyperparams.labelAlpha.from_string("root");
	int tmpID2 = m_driver._hyperparams.labelAlpha.from_string("ROOT");
	if (tmpID1 >= 0)
		m_driver._hyperparams.root = "root";
	if (tmpID2 >= 0)
		m_driver._hyperparams.root = "ROOT";
	m_driver._hyperparams.rootID = m_driver._hyperparams.labelAlpha.from_string(m_driver._hyperparams.root);
	cout << "Word Num: " << m_driver._hyperparams.wordAlpha.m_size << endl;
	cout << "Root ID: " << m_driver._hyperparams.rootID << endl;
}
// get gold actions of instances and static action
void DepParser::getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions) {
	vecActions.clear();
	vecActions.resize(vecInsts.size());
	int instSize = vecInsts.size();
	vector<CState> all_states(max_length);
	CAction ac;
	CResult result;
	Metric uas, las, las_punc; 
	uas.reset();
	las.reset();
	las_punc.reset();
	int step;
	unordered_map<string, int> action_stat;
	for (int idx = 0; idx < instSize; idx++) {
		const Instance& inst = vecInsts[idx];
		vector<CAction>& actions = vecActions[idx];
		all_states[0].clear(); // clear first state;
		all_states[0].ready(&inst); //  the state of instance
		step = 0;
		while (!all_states[step].isEnd()) {
			all_states[step].getGoldAction(m_driver._hyperparams, inst.result, ac);
			actions.push_back(ac);
			action_stat[ac.str(m_driver._hyperparams)]++;
			all_states[step].move(&all_states[step + 1], ac);
			step++;
		}
		all_states[step].getResults(result, m_driver._hyperparams);
		inst.evaluate(result, uas, las, las_punc); 
		assert(uas.bIdentical() && las.bIdentical() && las_punc.bIdentical()); // check the actions

		if ((idx + 1) % m_options.verboseIter == 0)
			cout << idx + 1 << " ";
	}
	cout << endl;
	m_driver._hyperparams.actionAlpha.initial(action_stat, 0);
	m_driver._hyperparams.actionAlpha.set_fixed_flag(true);
	m_driver._hyperparams.actionNum = m_driver._hyperparams.actionAlpha.size();
	cout << "Action num: " << m_driver._hyperparams.actionNum << endl;
}

void DepParser::train(const string &trainFile, const string &devFile, const string &testFile, const string &modelFile,
	const string &optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);

	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
	}
	createAlphabet(trainInsts);
	vector<vector<CAction> > trainInstGoldActions;
	getGoldActions(trainInsts, trainInstGoldActions);

	// pretrain
	m_driver._modelparams.extWordEmb.initial(&m_driver._hyperparams.extWordAlpha, m_options.wordEmbFile, false);
	// random
	m_driver._modelparams.wordEmb.initial(&m_driver._hyperparams.wordAlpha, m_options.wordEmbSize, true);

	m_driver._hyperparams.extWordDim = m_driver._modelparams.extWordEmb.nDim;
	m_driver._hyperparams.wordDim = m_driver._modelparams.wordEmb.nDim;
	m_driver._hyperparams.wordRepresentDim = m_driver._hyperparams.extWordDim + m_driver._hyperparams.wordDim;

	m_driver._modelparams.actionEmb.initial(&m_driver._hyperparams.actionAlpha, m_options.actionEmbSize, true);
	m_driver._hyperparams.actionDim = m_driver._modelparams.actionEmb.nDim;

	m_driver._hyperparams.setRequared(m_options);
	m_driver.initial();

	int inputSize = trainInsts.size();
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);
	int devNum = devInsts.size(), testNum = testInsts.size();
	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;
	vector<Instance> subInstances;
	vector<vector<CAction> > subGoldActions;
	Metric eval;
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter + 1 << std::endl;
		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;

		for (int idx = 0; idx < batchBlock; idx++) {
			int start_pos = idx * m_options.batchSize;
			int end_pos = (idx + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;
			subInstances.clear();
			subGoldActions.clear();
			auto t_start = std::chrono::high_resolution_clock::now();
			for (int idy = start_pos; idy < end_pos; idy++) { // one batch
				subInstances.push_back(trainInsts[indexes[idy]]);
				subGoldActions.push_back(trainInstGoldActions[indexes[idy]]);
			}
			dtype cost = m_driver.train(subInstances, subGoldActions);
			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;
			if ((idx + 1) % (m_options.verboseIter) == 0) {
				auto t_end = std::chrono::high_resolution_clock::now();
				std::cout << "current: " << idx + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
					<< ", Time: " << std::chrono::duration<double>(t_end - t_start).count() << "s" << std::endl;
			}
			m_driver.updateModel();
		}
		bool bCurIterBetter;
		vector<CResult> decodeInstResults;
		Metric dev_uas, dev_las, dev_las_punc;
		Metric test_uas, test_las, test_las_punc;
		double bestFmeasure = -1;

		if (devNum > 0) {
			auto t_start_dev = std::chrono::high_resolution_clock::now();
			cout << "Dev start." << std::endl;
			bCurIterBetter = false;
			if (!m_options.outBest.empty()) {
				decodeInstResults.clear();
			}
			dev_uas.reset();
			dev_las.reset();
			dev_las_punc.reset();
			predict(devInsts, decodeInstResults);
			int devNum = devInsts.size();
			for (int idx = 0; idx < devNum; idx++) {
				devInsts[idx].evaluate(decodeInstResults[idx], dev_uas, dev_las, dev_las_punc);
			}
			auto t_end_dev = std::chrono::high_resolution_clock::now();
			cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(t_end_dev - t_start_dev).count() << std::endl;
			cout << "dev:" << std::endl;
			dev_uas.print();
			dev_las.print();
			dev_las_punc.print();
			if (!m_options.outBest.empty() && dev_las.getAccuracy() > bestFmeasure) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				auto t_start_test = std::chrono::high_resolution_clock::now();
				cout << "Test start." << std::endl;
				if (!m_options.outBest.empty()) {
					decodeInstResults.clear();
				}
				test_uas.reset(); 
				test_las.reset();
				test_las_punc.reset();
				predict(testInsts, decodeInstResults);
				for (int idx = 0; idx < testInsts.size(); idx++) {
					testInsts[idx].evaluate(decodeInstResults[idx], test_uas, test_las, test_las_punc);
				}
				auto t_end_test = std::chrono::high_resolution_clock::now();
				cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
				cout << "test:" << std::endl;
				test_uas.print();
				test_las.print();
				test_las_punc.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}
		}

	}
}

void DepParser::test(const string &testFile, const string &outputFile, const string &modelFile) {
}

void DepParser::writeModelFile(const string &outputModelFile) {
}

void DepParser::loadModelFile(const string &inputModelFile) {
}

void DepParser::predict(const vector<Instance> &input, vector<CResult> &output) {
	vector<Instance> batch_input;
	vector<CResult> batch_output;
	int input_size = input.size();
	output.clear();
	for (int idx = 0; idx < input_size; idx++)
	{
		batch_input.push_back(input[idx]);
		if (batch_input.size() == m_options.batchSize || idx == input_size - 1) {
			batch_output.clear();
			m_driver.decode(batch_input, batch_output);
			batch_input.clear();
			output.insert(output.end(), batch_output.begin(), batch_output.end());
		}
	}
}
int main(int argc, char* argv[]) {
	std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
	std::string optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;
	int threads = 2;


	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	ah.new_named_int("th", "thread", "named_int", "number of threads for openmp", threads);

	ah.process(argc, argv);

	DepParser parser;
	if (bTrain) {
		parser.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		parser.test(testFile, outputFile, modelFile);
	}
}
