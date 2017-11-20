#ifndef BASIC_INSTANCE_READER_H
#define BASIC_INSTANCE_READER_H

#include "Reader.h"
#include "N3LDG.h"
#include <sstream>

class InstanceReader : public Reader {
public:
  InstanceReader() {
  }

  ~InstanceReader() {
  }

  Instance *getNext() {
    m_instance.clear();
    static string strLine;
    static vector<string> vecLine;
    vecLine.clear();
    while (1) {
      if (!my_getline(m_inf, strLine)) {
        break;
      }
      if (strLine.empty())
        break;
      vecLine.push_back(strLine);
    }

    static vector<string> charInfo;
    static vector<string> tmpInfo;
    static int count, parent_id;
    count = 0;
    for (int i = 0; i < vecLine.size(); i++) {
      split_bychar(vecLine[i], charInfo, '\t');
      m_instance.words.push_back(charInfo[0]);
			m_instance.tags.push_back(charInfo[1]);
      m_instance.result.tags.push_back(charInfo[1]);
      m_instance.result.heads.push_back(atoi(charInfo[2].c_str()));
      m_instance.result.labels.push_back(charInfo[3]);

      // check projectivity
      int head = atoi(charInfo[2].c_str());
      int mini = i < head ? i : head;
      int maxi = i > head ? i : head;
      for (int j = mini + 1; j < maxi; ++j) {
        split_bychar(vecLine[j], tmpInfo, '\t');
        if (atoi(tmpInfo[2].c_str()) < mini || atoi(tmpInfo[2].c_str()) > maxi) {
          std::cout << "Non-projective sentence found, skipped. \n";
          return getNext();
        }
      }
    }

    m_instance.result.words = m_instance.words;

    return &m_instance;
  }
};

#endif

