#include <utility>

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <iostream>
#include <string>

/*
 *  Simple wrapper around the std::chrono functions which will be used for
 *  timing.
 */

class Timer {
 private:
  std::chrono::time_point<std::chrono::system_clock> _start;  // start time
  std::chrono::time_point<std::chrono::system_clock> _end;    // end time
  long long _duration;  // end - start, i.e. for how long the timer was running
  std::string _title;   // "name" of the timer, displayed with the elapsed time
  bool _stopped;

 public:
  // constructor
  Timer(std::string title)
      : _title(std::move(title)),  // the title will be temporary string anyway,
                                   // so no need to pass it by reference
        _stopped(false),
        _duration(0) {
    // might as well start the timer when the object is constructed
    // to save extra typing
    start();
  }

  // starts the timer
  void start() {
    // get current time
    _start = std::chrono::system_clock::now();
  }

  // stops the timer and displays the elapsed time
  void stop() {
    // get current time
    _end = std::chrono::system_clock::now();
    // calculate duration
    _duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start)
            .count();
    // display it
    std::cout << _duration << "\n" << std::flush;
    // remember that the timer has been stop explicitly
    _stopped = true;
  }

  // returns the duration
  long long duration() const { return _duration; }

  // stops the timer if it hasn't been stopped when
  // the object goes out of scope, i.e. when it's destroyed
  ~Timer() {
    if (!_stopped) stop();
  }
};

#endif  // TIMER_H_
