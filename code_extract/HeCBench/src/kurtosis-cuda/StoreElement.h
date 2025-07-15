#ifndef STOREELEMENT_H_
#define STOREELEMENT_H_

#include <string>
#include <sstream>
#include <cstring>
#
#include "StoreTypedefs.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

/*
  storeElement is the data structure send to this system by customers as a single
  time series event. It contains time as unsigned long long int, is tagged, has
  a value and belongs to certain series.
*/
struct storeElement
{
  public:
    /* FIELDS */
    int32_t tag;
    metric_type metric;
    ullint time;
    store_value_type value;

    /* CONSTRUCTORS */

    HOST DEVICE
      storeElement(){ tag = 0; metric = 0; time = 0; value = 0; }
    HOST DEVICE
      storeElement(int _tag, metric_type _metric, ullint _time, store_value_type _value)
      : tag(_tag), metric(_metric), time(_time), value(_value) {}
    HOST DEVICE
      storeElement(const storeElement& elem)
      {
        this->tag = elem.tag;
        this->metric = elem.metric;
        this->time = elem.time;
        this->value = elem.value;
      }
    HOST DEVICE
      ~storeElement(){}

    HOST DEVICE
      storeElement & operator= (const storeElement& elem)
      {
        tag = elem.tag;
        metric = elem.metric;
        time = elem.time;
        value = elem.value;
        return *this;
      }

    std::string toString()
    {
      std::ostringstream stream;
      stream << "[" << tag << ", " << metric << ", " << time << ", " << value << "]";
      return  stream.str();
    }
};


#endif /* defined( STOREELEMENT_H_) */
