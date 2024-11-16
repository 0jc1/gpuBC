/*
---------------------------------------------------------------------
 This file is a part of the source code for the paper "Betweenness
 Centrality on GPUs and Heterogeneous Architectures", published in
 GPGPU'13 workshop. If you use the code, please cite the paper.

 Copyright (c) 2013,
 By:    Ahmet Erdem Sariyuce,
        Kamer Kaya,
        Erik Saule,
        Umit V. Catalyurek
---------------------------------------------------------------------
 This file is licensed under the Apache License. For more licensing
 information, please see the README.txt and LICENSE.txt files in the
 main directory.
---------------------------------------------------------------------
*/

#ifndef TIMESTAMP_ERIK_H__
#define TIMESTAMP_ERIK_H__

#include <sys/time.h>
#include <iostream>
#include <stdio.h>


namespace util{class timestamp;}

static std::ostream& operator<< (std::ostream&, const util::timestamp&);

namespace util
{
  /**
   * @brief allow to keep track of time with a precision of a
   * microseconds.
   *
   * The default constructors obtains the time with
   * gettimeofday(2). Timestamps can be added or substracted and
   * pushed to streams such as std::cout. It keeps track of time using
   * integers so as not to lose precision. Typical use of the class
   * are: \code 
   * timestamp t1; 
   * foo();
   * timestamp t2;
   * std::cout<<"foo() took "<<t2-t1<<" seconds"<<std::endl;
   * \endcode
   * or
   * \code
   * timestamp acc (0,0);
   * while (some_condition()){ 
   *   do_stuff();
   *   timestamp t1;
   *   do_important_stuff();
   *   timestamp t2;
   *   acc += t2-t1;
   *   do_more_stuff();
   * }
   * std::cout<<"do_important_stuff() took "<<acc<<" seconds"<<std::endl;
   * \endcode
   **/
  class timestamp
  {
    int seconds;
    int microseconds; //< 1000000

    /** @brief fix overflow after an add or a substract.
     *
     * Notice it only fix the overflow of one unit.
     **/
    void fixme()
    {
      if (microseconds < 0)
	{
	  seconds --;
	  microseconds += 1000000;
	}
      else if (microseconds > 1000000)
	{
	  seconds ++;
	  microseconds -= 1000000;
	}    
    }

  public:
    /// obtains time using gettimeofday(2)
    timestamp()
    {
      struct timeval tv;
      gettimeofday (&tv, NULL); //should check return code
      seconds = tv.tv_sec;
      microseconds = tv.tv_usec;
    }

    /// provides time in seconds and microseconds
    timestamp(int s, int us)
      :seconds(s), microseconds(us)
    {}

    timestamp(const timestamp& m)
      :seconds(m.seconds), microseconds(m.microseconds)
    {      
    }

    timestamp operator+ (const timestamp& b)
    {
      timestamp ret (this->seconds + b.seconds,
		     this->microseconds+b.microseconds);
      ret.fixme();
      return ret;
    }

    timestamp operator- (const timestamp& b)
    {
      timestamp ret (this->seconds - b.seconds,
		     this->microseconds - b.microseconds);
      ret.fixme();
      return ret;
    }


    operator double () const
    {
      double ret = this->seconds + this->microseconds/1000000.;
      return ret;
    }


    const timestamp& operator += (const timestamp& b)
    {
      this->seconds += b.seconds;
      this->microseconds += b.microseconds;

      this->fixme();
      return *this;
    }

    const timestamp& operator -= (const timestamp& b)
    {
      this->seconds -= b.seconds;
      this->microseconds -= b.microseconds;

      this->fixme();
      return *this;
    }

    friend std::ostream& ::operator<< (std::ostream&, const util::timestamp&);
    
    timestamp operator/ (const int i) const
    {
      timestamp t (this->seconds / i, (1000000*(this->seconds%i)+this->microseconds)/i);
      
      t.fixme();
      return t;
    }

    const timestamp& operator/= (const int i)
    {
      //do not invert the following two lines!
      this->microseconds = (1000000*(this->seconds%i)+this->microseconds)/i;
      this->seconds = this->seconds/i;
      
      this->fixme(); //I don't think that fixem() is necessary
      return *this;
    }

    void print ()
    {
      printf ("%d.%06d",seconds, microseconds);
    }

    void to_c_str(char* out, int size)
    {
      snprintf (out, size, "%d.%06d",seconds, microseconds);
    }
  };

}

static std::ostream& operator<< (std::ostream& out, const util::timestamp& t)
{
  char microsecval[7];

  out<<t.seconds<<'.';

  sprintf(microsecval, "%06d", t.microseconds);

  out<<microsecval;

  return out;
}


#endif
