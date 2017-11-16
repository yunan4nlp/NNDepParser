#ifndef CAction_H_
#define CAction_H_

class CAction {
public:
	enum CODE { ARC_LEFT = 0, ARC_RIGHT = 1, SHIFT = 2, POP_ROOT = 3, NO_ACTION = 4 };
	short _label; 
	unsigned long _code;

	inline bool isNone() const { return _code == NO_ACTION; }

	inline bool isFinish() const { return _code == POP_ROOT; }

	inline bool isShift() const { return _code == SHIFT; }

	inline bool isArcLeft() const { return _code == ARC_LEFT; }

	inline bool isArcRight() const { return _code == ARC_RIGHT; }

	inline std::string str(const HyperParams &opts) const {
		if (isShift())
			return "SHIFT";
		else if (isArcLeft())
			return "ARC_LEFT_" + opts.labelAlpha.from_id(_label);
		else if (isArcRight())
			return "ARC_RIGHT_" + opts.labelAlpha.from_id(_label);
		else if (isFinish())
			return "POP_ROOT";
		else
			return "NO_ACTION";
	}

	inline void clear() {
		_code = NO_ACTION;
		_label = -1;
	}

	inline void set(const int &code, const short &label) {
		_code = code;
		_label = label;
	}

	inline void set(const CAction &ac) {
		_code = ac._code;
		_label = ac._label;
	}

	bool operator == (const CAction &action) const {
		return _code == action._code && _label == action._label;
	}

	bool operator != (const CAction &action) const {
		return _code != action._code || _label != action._label;
	}
};


#endif /* CAction_H_ */