/*!
 * Copyright (c) 2016 by Contributors
 * \file optional.h
 * \brief Container to hold optional data.
 */
#ifndef DMLC_OPTIONAL_H_
#define DMLC_OPTIONAL_H_

#include <iostream>
#include <string>
#include <utility>
#include <algorithm>

#include "./base.h"
#include "./common.h"
#include "./logging.h"
#include "./type_traits.h"

namespace dmlc {

/*! \brief dummy type for assign null to optional */
struct nullopt_t {
#if defined(_MSC_VER) && _MSC_VER < 1900
  /*! \brief dummy constructor */
  explicit nullopt_t(int a) {}
#else
  /*! \brief dummy constructor */
  constexpr explicit nullopt_t(int a) {}
#endif
};

/*! Assign null to optional: optional<T> x = nullopt; */
constexpr const nullopt_t nullopt = nullopt_t(0);
/*! \brief C++14 aliases */
template <typename T> using decay_t = typename std::decay<T>::type;
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/*! \brief disambiguation tags that can be passed to the constructors of
 * dmlc::optional. A tag type to tell constructor to construct its value in-place
 */
struct in_place_t {
  explicit in_place_t() = default;  // NOLINT(*)
};
/*! \brief A tag to tell constructor to construct its value in-place */
static constexpr in_place_t in_place{};

/*! \brief  Is not constructible or convertible from any expression of type
(possibly const) dmlc::optional<U>, i.e., the following 8 type traits are all
false: Link: https://en.cppreference.com/w/cpp/utility/optional/optional */
template <typename T> class optional;
template <typename T, typename U, typename Other>
using enable_constructor_from_other =
    enable_if_t<std::is_constructible<T, Other>::value &&
                !std::is_constructible<T, optional<U> &>::value &&
                !std::is_constructible<T, optional<U> &&>::value &&
                !std::is_convertible<optional<U> &, T>::value &&
                !std::is_convertible<optional<U> &&, T>::value &&
                !std::is_constructible<T, const optional<U> &>::value &&
                !std::is_constructible<T, const optional<U> &&>::value &&
                !std::is_convertible<const optional<U> &, T>::value &&
                !std::is_convertible<const optional<U> &&, T>::value>;
template <typename T, typename U>
using enable_constructor_from_value =
    enable_if_t<std::is_constructible<T, U &&>::value &&
                !std::is_same<decay_t<U>, in_place_t>::value &&
                !std::is_same<optional<T>, decay_t<U>>::value>;
/*!
 * \brief c++17 compatible optional class.
 *
 * At any time an optional<T> instance either
 * hold no value (string representation "None")
 * or hold a value of type T.
 */
template <typename T>
class optional {
 public:
  using value_type = T;
  /*! \brief constructs an object that does not contain a value. */
  optional() : is_none(true) {}
  /*! \brief constructs an object that does contain a nullopt value. */
  optional(dmlc::nullopt_t) noexcept : is_none(true) {} // NOLINT(*)
  /*! \brief construct an optional object with another optional object */
  optional(const optional<T>& other) {
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    is_none = other.is_none;
    if (!is_none) {
      new (&val) T(other.value());
    }
#pragma GCC diagnostic pop
  }
  /*! \brief move constructor: If other contains a value, then stored value is
   * direct-intialized with it. */
  optional(optional &&other) noexcept(
      std::is_nothrow_move_constructible<T>::value
          &&std::is_nothrow_move_assignable<T>::value) {
    if (!other.has_value()) {
      reset();
    } else if (has_value()) {
      **this = std::move(*other);
    } else {
      new (&val) T(std::move(*other));
      is_none = false;
    }
  }
  /*! \brief constructs an optional object that contains a value, initialized as
   * if direct-initializing */
  template <typename... Args>
  explicit optional(
      enable_if_t<std::is_constructible<T, Args...>::value, in_place_t>,
      Args &&...args) {
    new (&val) T(std::forward<Args>(args)...);
    is_none = false;
  }
  /*! \brief Constructs an optional object that contains a value, initialized as
   * if direct-initializing */
  template <typename U, typename... Args>
  explicit optional(
      enable_if_t<std::is_constructible<T, std::initializer_list<U> &,
                                        Args &&...>::value,
                  in_place_t>,
      std::initializer_list<U> ilist, Args &&...args) {
    construct(ilist, std::forward<Args>(args)...);
  }
  /*! \brief constructs the stored value with value with `other` parameter.*/
  template <typename U = value_type,
            enable_if_t<std::is_convertible<U &&, T>::value> * = nullptr,
            enable_constructor_from_value<T, U> * = nullptr>
  optional(U &&other) noexcept {  // NOLINT(*)
    new (&val) T(std::forward<U>(other));
    is_none = false;
  }
  /*! \brief explicit constructor: constructs the stored value with `other`
   * parameter. */
  template <typename U = value_type,
            enable_if_t<!std::is_convertible<U &&, T>::value> * = nullptr,
            enable_constructor_from_value<T, U> * = nullptr>
  explicit optional(U &&other) noexcept {
    new (&val) T(std::forward<U>(other));
    is_none = false;
  }
  /*! \brief converting copy constructor */
  template <typename U,
            enable_constructor_from_other<T, U, const U &> * = nullptr,
            enable_if_t<std::is_convertible<const U &, T>::value> * = nullptr>
  optional(const optional<U> &other) {
    if (other.has_value()) {
      construct(*other);
    }
  }
  /*! \brief explicit converting copy constructor */
  template <typename U,
            enable_constructor_from_other<T, U, const U &> * = nullptr,
            enable_if_t<!std::is_convertible<const U &, T>::value> * = nullptr>
  explicit optional(const optional<U> &other) {
    if (other.has_value()) {
      construct(*other);
    }
  }
  /*! \brief converting move constructor */
  template <typename U, enable_constructor_from_other<T, U, U &&> * = nullptr,
            enable_if_t<std::is_convertible<U &&, T>::value> * = nullptr>
  optional(optional<U> &&other) {
    if (other.has_value()) {
      construct(std::move(*other));
    }
  }
  /*! \brief explicit converting move constructor */
  template <typename U, enable_constructor_from_other<T, U, U &&> * = nullptr,
            enable_if_t<!std::is_convertible<U &&, T>::value> * = nullptr>
  explicit optional(optional<U> &&other) {
    if (other.has_value()) {
      construct(std::move(*other));
    }
  }
  /*! \brief reset */
  void reset() noexcept {
    if (!has_value())
      return;
    (**this).T::~T();
    is_none = true;
  }
  /*! \brief deconstructor */
  ~optional() {
    if (!is_none) {
      reinterpret_cast<T*>(&val)->~T();
    }
  }
  /*! \brief swap two optional */
  void swap(optional<T>& other) {
    std::swap(val, other.val);
    std::swap(is_none, other.is_none);
  }
  /*! \brief set this object to hold value
   *  \param value the value to hold
   *  \return return self to support chain assignment
   */
  optional<T>& operator=(const T& value) {
    (optional<T>(value)).swap(*this);
    return *this;
  }
  /*! \brief set this object to hold the same value with other
   *  \param other the other object
   *  \return return self to support chain assignment
   */
  optional<T>& operator=(const optional<T> &other) {
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    (optional<T>(other)).swap(*this);
    return *this;
#pragma GCC diagnostic pop
  }
  /*! \brief clear the value this object is holding.
   *         optional<T> x = nullopt;
   */
  optional<T>& operator=(nullopt_t) {
    (optional<T>()).swap(*this);
    return *this;
  }
  /*! \brief non-const dereference operator */
  T& operator*() {  // NOLINT(*)
    return *reinterpret_cast<T*>(&val);
  }
  /*! \brief const dereference operator */
  const T& operator*() const {
    return *reinterpret_cast<const T*>(&val);
  }
  /*! \brief equal comparison */
  bool operator==(const optional<T>& other) const {
    return this->is_none == other.is_none &&
           (this->is_none == true || this->value() == other.value());
  }
  /*! \brief return the holded value.
   *         throws std::logic_error if holding no value
   */
  T &value() & {
    if (is_none) {
      throw std::logic_error("bad optional access");
    }
    return *reinterpret_cast<T *>(&val);
  }
  const T &value() const & {
    if (is_none) {
      throw std::logic_error("bad optional access");
    }
    return *reinterpret_cast<const T *>(&val);
  }
  T &&value() && {
    if (is_none) {
      throw std::logic_error("bad optional access");
    }
    return std::move(value());
  }
  const T &&value() const && {
    if (is_none) {
      throw std::logic_error("bad optional access");
    }
    return std::move(value());
  }
  /*! \brief whether this object is holding a value */
  explicit operator bool() const { return !is_none; }
  /*! \brief whether this object is holding a value (alternate form). */
  bool has_value() const { return operator bool(); }

 private:
  // whether this is none
  bool is_none = true;
  // on stack storage of value
  typename std::aligned_storage<sizeof(T), alignof(T)>::type val;

  template <typename... Args> void construct(Args &&...args) noexcept {
    new (std::addressof(val)) T(std::forward<Args>(args)...);
    is_none = false;
  }
};

/*! \brief serialize an optional object to string.
 *
 *  \code
 *    dmlc::optional<int> x;
 *    std::cout << x;  // None
 *    x = 0;
 *    std::cout << x;  // 0
 *  \endcode
 *
 *  \param os output stream
 *  \param t source optional<T> object
 *  \return output stream
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const optional<T> &t) {
  if (t) {
    os << *t;
  } else {
    os << "None";
  }
  return os;
}

/*! \brief parse a string object into optional<T>
 *
 *  \code
 *    dmlc::optional<int> x;
 *    std::string s1 = "1";
 *    std::istringstream is1(s1);
 *    s1 >> x;  // x == optional<int>(1)
 *
 *    std::string s2 = "None";
 *    std::istringstream is2(s2);
 *    s2 >> x;  // x == optional<int>()
 *  \endcode
 *
 *  \param is input stream
 *  \param t target optional<T> object
 *  \return input stream
 */
template<typename T>
std::istream &operator>>(std::istream &is, optional<T> &t) {
  char buf[4];
  std::streampos origin = is.tellg();
  is.read(buf, 4);
  if (is.fail() || buf[0] != 'N' || buf[1] != 'o' ||
      buf[2] != 'n' || buf[3] != 'e') {
    is.clear();
    is.seekg(origin);
    T x;
    is >> x;
    t = x;
    if (std::is_integral<T>::value && !is.eof() && is.peek() == 'L') is.get();
  } else {
    t = nullopt;
  }
  return is;
}
/*! \brief specialization of '>>' istream parsing for optional<bool>
 *
 * Permits use of generic parameter FieldEntry<DType> class to create
 * FieldEntry<optional<bool>> without explicit specialization.
 *
 *  \code
 *    dmlc::optional<bool> x;
 *    std::string s1 = "true";
 *    std::istringstream is1(s1);
 *    s1 >> x;  // x == optional<bool>(true)
 *
 *    std::string s2 = "None";
 *    std::istringstream is2(s2);
 *    s2 >> x;  // x == optional<bool>()
 *  \endcode
 *
 *  \param is input stream
 *  \param t target optional<bool> object
 *  \return input stream
 */
inline std::istream &operator>>(std::istream &is, optional<bool> &t) {
  // Discard initial whitespace
  while (isspace(is.peek()))
    is.get();
  // Extract chars that might be valid into a separate string, stopping
  // on whitespace or other non-alphanumerics such as ",)]".
  std::string s;
  while (isalnum(is.peek()))
    s.push_back(is.get());

  if (!is.fail()) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "1" || s == "true")
      t = true;
    else if (s == "0" || s == "false")
      t = false;
    else if (s == "none")
      t = nullopt;
    else
      is.setstate(std::ios::failbit);
  }

  return is;
}

/*! \brief description for optional int */
DMLC_DECLARE_TYPE_NAME(optional<int>, "int or None");
/*! \brief description for optional bool */
DMLC_DECLARE_TYPE_NAME(optional<bool>, "boolean or None");
/*! \brief description for optional float */
DMLC_DECLARE_TYPE_NAME(optional<float>, "float or None");
/*! \brief description for optional double */
DMLC_DECLARE_TYPE_NAME(optional<double>, "double or None");

}  // namespace dmlc

namespace std {
/*! \brief std hash function for optional */
template<typename T>
struct hash<dmlc::optional<T> > {
  /*!
   * \brief returns hash of the optional value.
   * \param val value.
   * \return hash code.
   */
  size_t operator()(const dmlc::optional<T>& val) const {
    std::hash<bool> hash_bool;
    size_t res = hash_bool(val.has_value());
    if (val.has_value()) {
      res = dmlc::HashCombine(res, val.value());
    }
    return res;
  }
};
}  // namespace std

#endif  // DMLC_OPTIONAL_H_
