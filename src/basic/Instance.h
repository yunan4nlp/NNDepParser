#ifndef BASIC_INSTANCE_H
#define BASIC_INSTANCE_H

#include "N3LDG.h"
#include "Metric.h"
#include "Result.h"

class Instance {
public:
	Instance() {
	}

	Instance(const Instance &other) {
		copyValuesFrom(other);
	}

	~Instance() {
	}

public:

	int size() const {
		return words.size();
	}


	void clear() {
		words.clear();
		result.clear();
		tags.clear();
	}

	void allocate(const int &size) {
		if (words.size() != size) {
			words.resize(size);
			tags.resize(size);
		}
		result.allocate(size);
	}


	void copyValuesFrom(const Instance &anInstance) {
		allocate(anInstance.size());
		for (int i = 0; i < anInstance.size(); i++) {
			words[i] = anInstance.words[i];
			tags[i] = anInstance.tags[i];
		}

		result.copyValuesFrom(anInstance.result, &words);
	}

	void evaluate(CResult &other, Metric &uas, Metric &las, Metric &las_punc) const {
		int sent_size = result.heads.size();
		uas.predicated_label_count = 0;
		las.predicated_label_count = 0;
		las_punc.predicated_label_count = 0;
		for (int idx = 0; idx < sent_size; idx++) {
			if (!isPunc(result.tags[idx])) {
				uas.overall_label_count++;
				las.overall_label_count++;
				if (result.heads[idx] == other.heads[idx]) {
					uas.correct_label_count++;
					if (result.labels[idx] == other.labels[idx])
						las.correct_label_count++;
				}
			}
			if (result.heads[idx] == other.heads[idx] && result.labels[idx] == other.labels[idx]) {
				las_punc.correct_label_count++;
			}
		}
		las_punc.overall_label_count += sent_size;
	}

public:
	vector<string> words;
	vector<string> tags;
	CResult result;
};

#endif
