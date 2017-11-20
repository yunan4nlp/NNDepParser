#ifndef ACTIONED_NODES_H 
#define ACTIONED_NODES_H

#include "ModelParams.h"
#include "AtomFeature.h"

// score the action one by one
class ActionedNodes {
public:
	LookupNode pre_action_input;
	LookupNode pre_pre_action_input;
	ConcatNode action_concat;
	IncLSTM1Builder action_lstm;

	vector<LookupNode> current_action_input;
	vector<PDotNode> action_score;
	vector<PAddNode> outputs;

	ConcatNode state_concat;
	UniNode state_hidden;
	BucketNode bucket_word;

	const HyperParams *pOpts;

	inline void initial(ModelParams &params, const HyperParams &hyparams) {
		current_action_input.resize(hyparams.actionNum);
		action_score.resize(hyparams.actionNum);
		outputs.resize(hyparams.actionNum);
		for (int idx = 0; idx < hyparams.actionNum; idx++) {
			current_action_input[idx].setParam(&params.scored_action_table);
			current_action_input[idx].init(hyparams.stateHiddenSize, hyparams.dropProb);
			action_score[idx].init(1, -1);
			outputs[idx].init(1, -1);
		}
		pre_action_input.setParam(&params.action_table);
		pre_action_input.init(hyparams.actionDim, hyparams.dropProb);
		pre_pre_action_input.setParam(&params.action_table);
		pre_pre_action_input.init(hyparams.actionDim, hyparams.dropProb);
		action_concat.init(hyparams.actionDim * 2, -1);
		action_lstm.init(&params.action_lstm_params, hyparams.dropProb);
		bucket_word.init(hyparams.rnnHiddenSize * 4, -1);
		state_concat.init(hyparams.stateConcatSize, -1);
		state_hidden.setParam(&params.state_hidden_params);
		state_hidden.init(hyparams.stateHiddenSize, hyparams.dropProb);
		pOpts = &hyparams;
	}

	inline void forward(Graph *cg, const vector<CAction> &actions,  AtomFeat &atomFeat) {
		int action_num = actions.size();
		pre_action_input.forward(cg, atomFeat._pre_action_str);
		pre_pre_action_input.forward(cg, atomFeat._pre_pre_action_str);
		action_concat.forward(cg, &pre_action_input, &pre_pre_action_input);
		action_lstm.forward(cg, &action_concat, atomFeat._pre_action_lstm);
		CAction ac;
		bucket_word.forward(cg, 0);
		PNode pword_lstm_buffer0 =
			atomFeat._next_index >= 0 ? &(*atomFeat._pword_lstm)[atomFeat._next_index] : (PNode)&bucket_word;

		PNode pword_lstm_stack_top0 =
			atomFeat._stack_top_0 >= 0 ? &(*atomFeat._pword_lstm)[atomFeat._stack_top_0] : (PNode)&bucket_word;

		PNode pword_lstm_stack_top1 =
			atomFeat._stack_top_1 >= 0 ? &(*atomFeat._pword_lstm)[atomFeat._stack_top_1] : (PNode)&bucket_word;

		PNode pword_lstm_stack_top2 =
			atomFeat._stack_top_2 >= 0 ? &(*atomFeat._pword_lstm)[atomFeat._stack_top_2] : (PNode)&bucket_word;

		vector<PNode> feats;


		feats.push_back(&action_lstm._hidden);
		feats.push_back(pword_lstm_buffer0);
		feats.push_back(pword_lstm_stack_top0);
		feats.push_back(pword_lstm_stack_top1);
		feats.push_back(pword_lstm_stack_top2);


		state_concat.forward(cg, feats);
		state_hidden.forward(cg, &state_concat);
		vector<PNode> sumNodes;
		for (int idx = 0; idx < action_num; idx++) {
			sumNodes.clear();
			ac.set(actions[idx]);
			const string &action = ac.str(*pOpts);
			current_action_input[idx].forward(cg, action);
			action_score[idx].forward(cg, &current_action_input[idx], &state_hidden);
			sumNodes.push_back(&action_score[idx]);
			outputs[idx].forward(cg, sumNodes);
		}
	}
	
};

#endif /*ACTIONED_NODES_H*/