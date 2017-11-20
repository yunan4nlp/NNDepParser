#ifndef ATOM_FEATURE_H 
#define ATOM_FEATURE_H

#include "ModelParams.h"

class AtomFeat {
public:
	vector<ConcatNode> *_pword_lstm;

	int _next_index;
	string _pre_action_str;
	string _pre_pre_action_str;

	IncLSTM1Builder *_pre_action_lstm;

	int _stack_top_0;
	int _stack_top_1;
	int _stack_top_2;
};
#endif /*ATOM_FEATURE_H */