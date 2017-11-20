#ifndef Driver_H_
#define Driver_H_

#include "N3LDG.h"
#include "HyperParams.h"
#include "ModelParams.h"
#include "ComputionGraph.h"
#include "Action.h"


class Driver {
public:
	Driver() {}

	~Driver() {}

public:
	vector<GraphBuilder> _builder;
	HyperParams _hyperparams;
	ModelParams _modelparams;
	Graph _encoderGraph;
	vector<Graph> _decoderGraphs;
	ModelUpdate _ada;
	Metric _eval;

	inline void initial() {
		if (!_hyperparams.bValid()) {
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams)) {
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_builder.resize(_hyperparams.batch);
		_decoderGraphs.resize(_hyperparams.batch);
		for (int idx = 0; idx < _hyperparams.batch; idx++) {
			_builder[idx].initial(_modelparams, _hyperparams);
		}
		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		_modelparams.exportModelParams(_ada);
		_hyperparams.print();
	}

public:
	inline void setUpdateParameters(const dtype &nnRegular, const dtype &adaAlpha, const dtype &adaEps) {
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

	inline dtype train(const vector<Instance> &sentences, const vector<vector<CAction> > &gold_actions) {
		_eval.reset();
		dtype cost = 0.0;
		int num = sentences.size();
		_encoderGraph.clearValue(true);
		for (int idx = 0; idx < num; idx++) {
			_builder[idx].encode(&_encoderGraph, sentences[idx]);
		}
		_encoderGraph.compute();
		for (int idx = 0; idx < num; idx++) {
			_decoderGraphs[idx].clearValue(true);
			_builder[idx].decode(&_decoderGraphs[idx], sentences[idx], &gold_actions[idx]);
			cost += loss_google(_builder[idx], num);
			_decoderGraphs[idx].backward();
			_eval.overall_label_count += gold_actions[idx].size();
		}
		_encoderGraph.backward();
		return cost;
	}

	inline dtype loss_google(const GraphBuilder &builder, int batch) {
		int maxstep = builder.outputs.size();
		if (maxstep == 0) return 1.0;
		//_eval.correct_label_count += maxstep;
		PNode pBestNode = NULL;
		PNode pGoldNode = NULL;
		PNode pCurNode;
		dtype sum, max;
		int curcount, goldIndex;
		vector<dtype> scores;
		dtype cost = 0.0;

		for (int step = 0; step < maxstep; step++) {
			curcount = builder.outputs[step].size();
			if (curcount == 1) {
				_eval.correct_label_count++;
				continue;
			}
			max = 0.0;
			goldIndex = -1;
			pBestNode = pGoldNode = NULL;
			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
					pBestNode = pCurNode;
				}
				if (builder.outputs[step][idx].is_gold) {
					pGoldNode = pCurNode;
					goldIndex = idx;
				}
			}

			if (goldIndex == -1) {
				std::cout << "impossible" << std::endl;
			}
			pGoldNode->loss[0] = -1.0 / batch;

			max = pBestNode->val[0];
			sum = 0.0;
			scores.resize(curcount);
			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				scores[idx] = exp(pCurNode->val[0] - max);
				sum += scores[idx];
			}

			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				pCurNode->loss[0] += scores[idx] / (sum * batch);
			}

			if (pBestNode == pGoldNode)_eval.correct_label_count++;
			//_eval.overall_label_count++;

			cost += -log(scores[goldIndex] / sum);

			if (std::isnan(cost)) {
				std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
			}

		}

		return cost;
	}

	inline void decode(const vector<Instance> &sentences, vector<CResult> &results){
		int step, num = sentences.size();
		if (num > _builder.size()) {
			std::cout << "input example number is larger than predefined batch number" << std::endl;
			return;
		}
		results.resize(num);
		_encoderGraph.clearValue();
		for (int idx = 0; idx < num; idx++) {
			_builder[idx].encode(&_encoderGraph, sentences[idx]);
		}

		_encoderGraph.compute();

		for (int idx = 0; idx < num; idx++) {
			_decoderGraphs[idx].clearValue();
			_builder[idx].decode(&_decoderGraphs[idx], sentences[idx]);
			step = _builder[idx].outputs.size();
			_builder[idx].states[step - 1].getResults(results[idx], _hyperparams);
		}
	}
	
	inline void updateModel() {
		_ada.updateAdam(_hyperparams.clips);
	}
	
	inline dtype loss(){
	}

};

#endif /* Driver_H_ */
