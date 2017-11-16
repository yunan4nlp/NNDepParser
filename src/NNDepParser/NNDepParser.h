#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "N3LDG.h"
#include "Options.h"
#include "Pipe.h"
#include "Driver.h"
#include "Utf.h"
#include "Action.h"
#include "State.h"

class DepParser {
public:
	DepParser();
	virtual ~DepParser();

public:
	Driver m_driver;
	Options m_options;
	Pipe m_pipe;

public:
	void createAlphabet(const vector<Instance> &vecInsts);

public:
	void train(const string &trainFile, const string &devFile, const string &testFile, const string &modelFile, const string &optionFile);
	void predict(const vector<Instance> &input, vector<CResult> &output);
	void test(const string &testFile, const string &outputFile, const string &modelFile);
	void getGoldActions(const vector<Instance> &vecInsts, vector<vector<CAction> > &vecActions);

public:
	void writeModelFile(const string &outputModelFile);
	void loadModelFile(const string &inputModelFile);
};



#endif /* SRC_PARSER_H_ */