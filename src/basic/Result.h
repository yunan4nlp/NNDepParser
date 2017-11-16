#ifndef BASIC_RESULT_H
#define BASIC_RESULT_H

#include <string>
#include <vector>
#include <fstream>
#include "N3LDG.h"
#include "Alphabet.h"
#include "Utf.h"


class CResult {
public:
  vector<string> tags;
  vector<int> heads;
  vector<string> labels;

  const vector<string> *words;

public:
  inline void clear() {
    words = nullptr;
    tags.clear();
    heads.clear();
    labels.clear();
  }

  inline void allocate(const int &size) {
    if (labels.size() != size) {
      tags.resize(size);
      heads.resize(size);
      labels.resize(size);

    }
  }

  inline int size() const {
    return heads.size();
  }

  inline void copyValuesFrom(const CResult &result) {
    static int size;
    size = result.size();
    allocate(size);

    for (int i = 0; i < size; i++) {
      tags[i] = result.tags[i];
      heads[i] = result.heads[i];
      labels[i] = result.labels[i];
    }
    words = result.words;
  }

  inline void copyValuesFrom(const CResult &result, const vector<string> *pwords) {
    static int size;
    size = result.size();
    allocate(size);

    for (int i = 0; i < size; i++) {
      tags[i] = result.tags[i];
      heads[i] = result.heads[i];
      labels[i] = result.labels[i];
    }
    words = pwords;
  }


  inline std::string str() const {
    for (int i = 0; i < size(); ++i) {
      std::cout << (*words)[i] << " " << tags[i] << " " << heads[i] << " " << labels[i] << std::endl;
    }
    std::cout << endl;
  }

};


#endif
