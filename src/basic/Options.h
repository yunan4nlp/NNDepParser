#ifndef BASIC_OPTIONS_H
#define BASIC_OPTIONS_H

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3LDG.h"

class Options {
public:
  int wordCutOff;
	int wordEmbSize;
	string wordEmbFile;

	int actionEmbSize;
	int hiddenSize;
	int rnnHiddenSize;
  int maxIter;
  int batchSize;
  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;
	dtype clips;
	dtype delta;

  int verboseIter;
  bool saveIntermediate;
	bool wordFineTune;
  int maxInstance;
  vector<string> testFiles;
  string outBest;

  int unkStrategy;

  Options() {
    wordCutOff = 0;
    maxIter = 1000;
    batchSize = 1;
    adaEps = 1e-6;
    adaAlpha = 0.001;
    regParameter = 1e-8;
    dropProb = -1;
		clips = 10;
		delta = 0;


    wordEmbSize = 100;
		wordFineTune = true;

		hiddenSize = 100;
		rnnHiddenSize = 100;

		actionEmbSize = 100;

    verboseIter = 100;
    saveIntermediate = true;
    maxInstance = -1;
    testFiles.clear();
    outBest = "";
    unkStrategy = 1;
  }

  virtual ~Options() {

  }

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());

      if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());

      if (pr.first == "clips")
        clips = atof(pr.second.c_str());

      if (pr.first == "delta")
        delta = atof(pr.second.c_str());

      if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbFile")
        wordEmbFile = pr.second;

      if (pr.first == "actionEmbSize")
        actionEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordFineTune")
				wordFineTune = (pr.second == "true") ? true : false;

      if (pr.first == "hiddenSize")
        hiddenSize = atoi(pr.second.c_str());
      if (pr.first == "rnnHiddenSize")
        rnnHiddenSize = atoi(pr.second.c_str());

      if (pr.first == "verboseIter")
        verboseIter = atoi(pr.second.c_str());
      if (pr.first == "saveIntermediate")
        saveIntermediate = (pr.second == "true") ? true : false;

      if (pr.first == "maxInstance")
        maxInstance = atoi(pr.second.c_str());
      if (pr.first == "testFile")
        testFiles.push_back(pr.second);
      if (pr.first == "outBest")
        outBest = pr.second;
    }
  }

  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;
    std::cout << "clips = " << clips << std::endl;
    std::cout << "delta = " << delta << std::endl;



    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "actionEmbSize = " << actionEmbSize << std::endl;

    std::cout << "wordEmbFile = " << wordEmbFile << std::endl;
    std::cout << "unkStrategy = " << unkStrategy << std::endl;



    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveIntermediate = " << saveIntermediate << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;
    for (int idx = 0; idx < testFiles.size(); idx++) {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }
    std::cout << "outBest = " << outBest << std::endl;

    std:cout << std::endl;
  }

  void load(const std::string &infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif
