#ifndef ATOM_FEATURE_H 
#define ATOM_FEATURE_H

#include "ModelParams.h"

class AtomFeat {
public:
	const LSTM1Builder* _pword_lstm_left;
	const LSTM1Builder* _pword_lstm_right;
	int _next_index;
};
#endif /*ATOM_FEATURE_H */