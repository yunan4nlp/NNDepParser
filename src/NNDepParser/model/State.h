#ifndef CState_H_
#define CState_H_

#include "ModelParams.h"
#include "AtomFeature.h"
#include "GlobalNodes.h"
#include "ActionedNodes.h"

class CState {
public:
	short _stack[max_length]; // the stack
	short _stack_size; // the size  of stack
	short _label[max_length]; // label
	short _head[max_length]; // head
	short _have_parent[max_length]; // recored head assigned

	short _next_index; // index of top element of buffer
	int _word_size; // word num of sentence

	const Instance* inst; // Instance 

	CState* _pre_state; //  previous action

	CAction _pre_action; //previous action
	bool _is_start; // first state or not
	bool _is_gold; // moving to gold action and being gold state
	AtomFeat _atom_feat; // atom feature of this state
	ActionedNodes _next_action_score; // score all candidate action
public:
	inline void initial(ModelParams &params, const HyperParams &hyparams) {
		_next_action_score.initial(params, hyparams);
	}
	
public:
	CState() {
		clear();
	}

	~CState(){
		clear();
	}

public:
	void clear() {
		_stack_size = 0;
		_next_index = 0;
		_word_size = 0;

		_is_start = true;
		_is_gold = true;
		_pre_state = NULL;
		doneMark();
	}

	void computeNextActionScore(Graph *cg, const vector<CAction> &candidate_actions){
		_next_action_score.forward(cg, candidate_actions, _atom_feat);
	}

	void getGoldAction(HyperParams &opts, const CResult &result, CAction &ac) {
		if (_stack_size == 0) {
			ac.set(CAction::SHIFT, -1);
		} else if (_stack_size == 1) {
			const int& top0 = _stack[_stack_size - 1];
			if (_next_index == _word_size)
				ac.set(CAction::POP_ROOT, opts.labelAlpha.from_string(result.labels[top0]));
			else
				ac.set(CAction::SHIFT, -1);
		} else if (_stack_size > 1) {
			const int& top0 = _stack[_stack_size - 1];
			const int& top1 = _stack[_stack_size - 2];
			if(result.heads[top1] == top0) { 
				ac.set(CAction::ARC_LEFT, opts.labelAlpha.from_string(result.labels[top1]));
			} else if (result.heads[top0] == top1) {
				// check the top of stack have right children or not.
				bool have_right_children = false;
				for (int idx = _next_index; idx < _word_size; idx++) {
					if (result.heads[idx] == top0) {
						have_right_children = true;
						break;
					}
				}
				if (have_right_children) {
					ac.set(CAction::SHIFT, -1);
				} else {
					ac.set(CAction::ARC_RIGHT, opts.labelAlpha.from_string(result.labels[top0]));
				}
			} else {
				ac.set(CAction::SHIFT, -1);
			}
		}
	}

	void getResults(CResult &result, const HyperParams &opts) const {
		assert(_word_size == _next_index);
		result.clear();
		result.allocate(_word_size);
		for (int idx = 0; idx < _word_size; idx++) {
			assert(_have_parent[idx] == 1); // check parent
			result.labels[idx] = opts.labelAlpha.from_id(_label[idx]);
			result.heads[idx] = _head[idx];
		}
		result.words = &inst->words;
	}

	// prepare instance
	inline void ready(const Instance *pInst) {
		this->inst = pInst;
		_word_size = inst->words.size();
	}

  // copy data to next state	
	void copyState(CState *next){
		memcpy(next->_stack, _stack, sizeof(short) * _stack_size);
		memcpy(next->_head, _head, sizeof(short) * _next_index);
		memcpy(next->_have_parent, _have_parent, sizeof(short) * _next_index);
		memcpy(next->_label, _label, sizeof(short) * _next_index);
		next->_word_size = _word_size;
		next->inst = inst;
	}

  // temp mark
	inline void doneMark() {
		_stack[_stack_size] = -2;
		_head[_next_index] = -2;
		_label[_next_index] = -2;
		_have_parent[_next_index] = -2;
	}
	
	// for all states, not for gold state
	bool allowShift() const {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		if (_next_index == _word_size) // _next_index  == _words_size  means buffer empty
			return false;
		else
			return true;
	}
	// for all states, not for gold state
	bool allowArcLeft() const {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		if (_stack_size > 1)
			return true;
		else
			return false;
	}
	// for all states, not for gold state
	bool allowSrcRight() const {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		if (_stack_size > 1)
			return true;
		else
			return false;
	}
	// for all states, not for gold state	
	bool allowPopRoot() const {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		if (_stack_size == 1 && _next_index == _word_size)
			return true;
		else
			return false;
	}

	// shift action, top of buffer -> stack
	void shift(CState *next) {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		next->_next_index = _next_index + 1;
		next->_stack_size = _stack_size + 1;
		copyState(next);

		next->_stack[next->_stack_size - 1] = _next_index;
		next->_have_parent[_next_index] = 0;
		_pre_state = this;
		next->doneMark();
		next->_pre_action.set(CAction::SHIFT, -1);
	}
	// left action
	void arcLeft(CState *next, const short &dep) {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		next->_next_index = _next_index;
		next->_stack_size = _stack_size - 1;
		copyState(next);

		int top0 = _stack[_stack_size - 1];
		int top1 = _stack[_stack_size - 2];
		next->_stack[next->_stack_size - 1] = top0;
		next->_head[top1] = top0;
		next->_have_parent[top1] = 1;
		next->_label[top1] = dep;

		_pre_state = this;
		next->doneMark();
		next->_pre_action.set(CAction::ARC_LEFT, dep);
	}
	//right action	
	void arcRight(CState *next, const short &dep) {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		next->_next_index = _next_index;
		next->_stack_size = _stack_size - 1;
		copyState(next);

		int top0 = _stack[_stack_size - 1];
		int top1 = _stack[_stack_size - 2];
		next->_head[top0] = top1;
		next->_have_parent[top0] = 1;
		next->_label[top0] = dep;

		_pre_state = this;
		next->doneMark();
		next->_pre_action.set(CAction::ARC_RIGHT, dep);
	}
  // pop root action
	void popRoot(CState *next, const short &dep) {
		assert(_next_index <= _word_size && _stack_size >= 0 && _word_size >= 0);
		next->_next_index = _word_size;
		next->_stack_size = 0;
		copyState(next);

		int top0 = _stack[_stack_size - 1];
		next->_head[top0] = -1;
		next->_have_parent[top0] = 1;
		next->_label[top0] = dep;

		_pre_state = this;
		next->doneMark();
		next->_pre_action.set(CAction::POP_ROOT, dep);
	}

	bool isEnd() const {
		if (_pre_action.isFinish())
			return true;
		else
			return false;
	}

	//move to next state 
	void move(CState *next, const CAction &ac) {
		next->_is_start = false; // if a state move to next, next state can't be start 
		next->_is_gold = false; // here we don't know the action is gold or not
		if (ac.isShift()) {
			shift(next);
		}
		else if (ac.isArcLeft()) {
			arcLeft(next, ac._label);
		}
		else if (ac.isArcRight()) {
			arcRight(next, ac._label);
		}
		else if (ac.isFinish()) {
			popRoot(next, ac._label); // pop root
		} else {
			cout << "error action" << endl;
			exit(0);
		}
	}
	// get candidate actions of this state.
	void getCandidateActions(vector<CAction>& actions, HyperParams& opts) const {
		actions.clear();
		CAction ac;

		if(isEnd()) {
			actions.push_back(ac);
			return;
		}

		if (allowShift()) {
			ac.set(CAction::SHIFT, -1);
			actions.push_back(ac);
		}

		if(allowArcLeft()) {
			int label_size = opts.labelAlpha.size();
			for (int idx = 0; idx < label_size; idx++) {
				ac.set(CAction::ARC_LEFT, idx);
				if (idx != opts.rootID && opts.actionAlpha.from_string(ac.str(opts)) >= 0) {
					actions.push_back(ac);
				}
			}
		}

		if (allowSrcRight()) {
			int label_size = opts.labelAlpha.size();
			for (int idx = 0; idx < label_size; idx++) {
				ac.set(CAction::ARC_RIGHT, idx);
				if (idx != opts.rootID && opts.actionAlpha.from_string(ac.str(opts)) >= 0) {
					actions.push_back(ac);
				}
			}
		}

		if(allowPopRoot()) {
			ac.set(CAction::POP_ROOT, opts.rootID);
			actions.push_back(ac);
		}
	}
	// prepare atom feature of this state
	void prepare(const GlobalNodes& globelnodes) {
		_atom_feat._pword_lstm_left = &(globelnodes.word_lstm_left);
		_atom_feat._pword_lstm_right = &(globelnodes.word_lstm_right);
		// next index is LSTM index
		_atom_feat._next_index = _next_index >= 0 && _next_index < _word_size ? _next_index : -1;
		//_atom_feat._next_index = _next_index;
	}
};


class CScoredAction {
public:
	CState* state; // current state
	dtype score; // the score of action

	CAction ac;
	bool is_gold; // is gold state AND move to gold action ? 
	int position; // the position of action.

	CScoredAction(){
		state = NULL;
		score = 0;
		is_gold = false;
		position = -1;
	}
public:
	bool operator <(const CScoredAction &a1) const {
		return score < a1.score;
	}
	bool operator >(const CScoredAction &a1) const {
		return score > a1.score;
	}
	bool operator <=(const CScoredAction &a1) const {
		return score <= a1.score;
	}
	bool operator >=(const CScoredAction &a1) const {
		return score >= a1.score;
	}
};

class CScoredActionCompare {
public:
	int operator()(const CScoredAction &o1, const CScoredAction &o2) const {
		if (o1.score < o2.score)
			return -1;
		else if (o1.score > o2.score)
			return 1;
		else
			return 0;
	}
};

class COutput {
public:
	PNode in;
	bool is_gold;

	COutput(){
		in = NULL;
		is_gold = false;
	}

	COutput(const COutput& output){
		in = output.in;
		is_gold = output.is_gold;
	}
};
#endif /* CState_H_ */