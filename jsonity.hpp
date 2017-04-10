/*

  JSonity : JSON Utility for C++   Version 1.1.5

  Copyright (c) 2014, Ichishino

  http://ichishino.nobody.jp/jsonity/index.html
  https://github.com/Ichishino/JSonity

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

*/

#ifndef JSONITY_HPP_
#define JSONITY_HPP_

// If using <std::list, set, deque, array, forward_list, ...>
// Include them before this file.

#include <map>
#include <vector>
#include <sstream>

//---------------------------------------------------------------------------//
// Platform
//---------------------------------------------------------------------------//

#if defined(_WINDOWS) || defined(_WIN32) || defined(_WIN64) 
#define JSONITY_OS_WINDOWS
#ifdef _MSC_VER
#define JSONITY_COMPILER_MSVC
#endif
#endif

#if !defined(JSONITY_OS_WINDOWS) || (_MSC_VER >= 1600) // TODO
#define JSONITY_SUPPORT_CXX_11
#endif

#ifdef JSONITY_SUPPORT_CXX_11
#include <cstdint>
#else
typedef int int32_t;
typedef unsigned int uint32_t;
typedef __int64 int64_t;
typedef unsigned short char16_t;
typedef uint32_t char32_t;
#endif

#ifdef JSONITY_COMPILER_MSVC
#define JSONITY_SNPRINTF  sprintf_s
#define JSONITY_FORM_I64   "%I64d"
#else
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#define JSONITY_SNPRINTF  snprintf
#define JSONITY_FORM_I64  ("%" PRId64)
#endif

#ifdef JSONITY_COMPILER_MSVC
#pragma warning (disable : 4503) // Disable truncated name warning
#endif

//---------------------------------------------------------------------------//
// Macro
//---------------------------------------------------------------------------//

#ifndef NDEBUG
#include <cassert>
#define JSONITY_ASSERT(exp) assert((exp))
#else
#define JSONITY_ASSERT
#endif

#define JSONITY_CHAR(ch) (static_cast<char_t>(ch))

#define JSONITY_THROW_TYPE_MISMATCH() \
    (throw TypeMismatchException(__LINE__))

#define JSONITY_TYPE_CHECK(exp) \
    { if (!(exp)) JSONITY_THROW_TYPE_MISMATCH(); }

#define JSONITY_VALUE_IMPL_STL_CONTAINER(type) \
    template<typename ValueType> \
    Value(const type<ValueType>& container) \
        {   assignArray(container);    } \
    template<typename ValueType> \
    void setArray(const type<ValueType>& container) \
        {   destroy(); assignArray(container);   } \
    template<typename ValueType> \
    Value& operator=(const type<ValueType>& container) \
        {   setArray(container); return *this;  }

#define JSONITY_VALUE_IMPL_STL_CONTAINER_FIXED_SIZE(type) \
    template<typename ValueType, size_t Size> \
    Value(const type<ValueType, Size>& container) \
        {   assignArray(container);    } \
    template<typename ValueType, size_t Size> \
    void setArray(const type<ValueType, Size>& container) \
        {   destroy(); assignArray(container);   } \
    template<typename ValueType, size_t Size> \
    Value& operator=(const type<ValueType, Size>& container) \
        {   setArray(container); return *this;  }

#define JSONITY_VALUE_OPERATOR_IOSTREAM(JsonType) \
    inline JsonType::IStream& operator>>( \
        JsonType::IStream& is, JsonType::Value& value) \
        {   if (!JsonType::decode(is, value)) \
                {   is.setstate(std::ios::failbit); } \
            return is;  } \
    inline JsonType::OStream& operator<<( \
        JsonType::OStream& os, const JsonType::Value& value) \
        {   JsonType::encode(value, os); return os; } \
    inline JsonType::OStream& operator<<( \
        JsonType::OStream& os, const JsonType::Array& arr) \
        {   JsonType::UnreadableEncodeContext ctx(os); \
            JsonType::encodeArray(ctx, arr); \
            return os;  } \
    inline JsonType::OStream& operator<<( \
        JsonType::OStream& os, const JsonType::Object& obj) \
        {   JsonType::UnreadableEncodeContext ctx(os); \
            JsonType::encodeObject(ctx, obj); \
            return os;  }

// namespace
namespace jsonity {

//---------------------------------------------------------------------------//
// JsonBase
//---------------------------------------------------------------------------//

template<typename CharType,
         typename CharTraitsType = std::char_traits<CharType>,
         typename CharAllocatorType = std::allocator<CharType> >
class JsonBase
{
public:
    template<typename UserValueType> class UserValue;
    class Value;
    class Cursor;
    class Error;
    class Exception;
    class TypeMismatchException;
    class EncodeStyle;
    class EncodeContext;

public:

    // basic char
    typedef CharType char_t;

    // String
    typedef std::basic_string<
        CharType, CharTraitsType, CharAllocatorType> String;

    // Array
    typedef std::vector<Value> Array;

    // Object
    typedef std::map<String, Value> Object;

    // Stream
    typedef std::basic_istream<
        CharType, CharTraitsType > IStream;
    typedef std::basic_ostream<
        CharType, CharTraitsType > OStream;
    typedef std::basic_istringstream<
        CharType, CharTraitsType,
        CharAllocatorType > IStringStream;
    typedef std::basic_ostringstream<
        CharType, CharTraitsType,
        CharAllocatorType > OStringStream;

private:

    class UserValueBase
    {
    public:
        virtual ~UserValueBase() {}

    protected:
        UserValueBase() {}

    private:
        virtual UserValueBase* duplicate() const = 0;
        virtual void encode(EncodeContext& ctx) const = 0;

        friend class Value;
    };

public:

    //-----------------------------------------------------------------------//
    // JsonBase::UserValue
    //-----------------------------------------------------------------------//

    template<typename UserValueType>
    class UserValue : public UserValueBase
    {
    public:
        virtual ~UserValue() {}

    protected:
        UserValue() {}

        // Encode
        virtual void encode(EncodeContext& ctx) const = 0;

    private:
        UserValueBase* duplicate() const
        {
            return new UserValueType(
                (const UserValueType&)*this);
        }
    };

    //-----------------------------------------------------------------------//
    // JsonBase::Value
    //-----------------------------------------------------------------------//

    class Value
    {
    public:
        typedef char Type;
        static const Type NullType = 0;
        static const Type NumberType = 1; 
        static const Type StringType = 2; 
        static const Type BooleanType = 3;
        static const Type RealType = 4; 
        static const Type ArrayType = 5;
        static const Type ObjectType = 6;
        static const Type UserType = 7;
        static const Type UserPtrType = 8;

        Value()
        {
            assignNull();
        }

        Value(int32_t number)
        {
            assignNumber(number);
        }

        Value(int64_t number)
        {
            assignNumber(number);
        }

        Value(const char_t* str)
        {
            assignString(str);
        }

        Value(const String& str)
        {
            assignString(str.c_str(), str.size());
        }

        Value(bool boolean)
        {
            assignBoolean(boolean);
        }

        Value(double real)
        {
            assignReal(real);
        }

        template<typename KeyType, typename ValueType>
        Value(const std::map<KeyType, ValueType>& map)
        {
            assignObject(map);
        }

        Value(const UserValueBase& userValue)
        {
            assignUserValue(userValue);
        }

        Value(UserValueBase* userValue)
        {
            assignUserValuePtr(userValue);
        }

        Value(const Value& other)
        {
            type_ = NullType;
            assignValue(other);
        }

#ifdef JSONITY_SUPPORT_CXX_11
        Value(Value&& other)
        {
            type_ = NullType;
            move(std::move(other));
        }
#endif

        ~Value()
        {
            destroy();
        }

        void destroy()
        {
            if (isString())
            {
                delete data_.str_;
                data_.str_ = NULL;
            }
            else if (isArray())
            {
                delete data_.arr_;
                data_.arr_ = NULL;
            }
            else if (isObject())
            {
                delete data_.obj_;
                data_.obj_ = NULL;
            }
            else if (isUserValue())
            {
                delete data_.user_;
                data_.user_ = NULL;
            }

            type_ = NullType;
        }

    public:

        JSONITY_VALUE_IMPL_STL_CONTAINER(std::vector);
#if defined(_LIST_) || defined(_GLIBCXX_LIST) || defined(_LIBCPP_LIST)
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::list);
#endif
#if defined(_DEQUE_) || defined(_GLIBCXX_DEQUE) || defined(_LIBCPP_DEQUE)
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::deque);
#endif
#if defined(_SET_) || defined(_GLIBCXX_SET) || defined(_LIBCPP_SET)
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::set);
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::multiset);
#endif
#if defined(_ARRAY_) || defined(_GLIBCXX_ARRAY) || defined(_LIBCPP_ARRAY)
        JSONITY_VALUE_IMPL_STL_CONTAINER_FIXED_SIZE(std::array);
#endif
#if defined(_FORWARD_LIST_) || defined(_GLIBCXX_FORWARD_LIST) || \
    defined(_LIBCPP_FORWARD_LIST)
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::forward_list);
#endif
#if defined(_UNORDERED_SET_) || defined(_GLIBCXX_UNORDERED_SET) || \
    defined(_LIBCPP_UNORDERED_SET)
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::unordered_set);
        JSONITY_VALUE_IMPL_STL_CONTAINER(std::unordered_multiset);
#endif

    public:

        // Type

        Type getType() const
        {
            return type_;
        }

        bool isNull() const
        {
            return (getType() == NullType);
        }

        bool isNumber() const
        {
            return (getType() == NumberType);
        }

        bool isString() const
        {
            return (getType() == StringType);
        }

        bool isBoolean() const
        {
            return (getType() == BooleanType);
        }

        bool isReal() const
        {
            return (getType() == RealType);
        }

        bool isArray() const
        {
            return (getType() == ArrayType);
        }

        bool isObject() const
        {
            return (getType() == ObjectType);
        }

        bool isUserValue() const
        {
            return (getType() == UserType);
        }

        bool isUserValuePtr() const
        {
            return (getType() == UserPtrType);
        }

    public:

        // Getter

        int64_t getNumber() const
        {
            if (isNumber())
            {
                return data_.n_;
            }
            else if (isReal())
            {
                return static_cast<int64_t>(data_.d_);
            }
            else if (isBoolean())
            {
                return (getBoolean() ? 1 : 0);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        String& getString()
        {
            JSONITY_TYPE_CHECK(isString());
            return *data_.str_;
        }

        const String& getString() const
        {
            JSONITY_TYPE_CHECK(isString());
            return *data_.str_;
        }

        bool getBoolean() const
        {
            if (isBoolean())
            {
                return data_.b_;
            }
            else if (isNumber())
            {
                return (getNumber() != 0);
            }
            else if (isReal())
            {
                return (getReal() != 0.0);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        double getReal() const
        {
            if (isReal())
            {
                return data_.d_;
            }
            else if (isNumber())
            {
                return static_cast<double>(data_.n_);
            }
            else if (isBoolean())
            {
                return (getBoolean() ? 1.0 : 0.0);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        Array& getArray()
        {
            JSONITY_TYPE_CHECK(isArray());
            return *data_.arr_;
        }

        const Array& getArray() const
        {
            JSONITY_TYPE_CHECK(isArray());
            return *data_.arr_;
        }

        Object& getObject()
        {
            JSONITY_TYPE_CHECK(isObject());
            return *data_.obj_;
        }

        const Object& getObject() const
        {
            JSONITY_TYPE_CHECK(isObject());
            return *data_.obj_;
        }

        template<typename UserValueType>
        UserValueType& getUserValue()
        {
            JSONITY_TYPE_CHECK(isUserValue() || isUserValuePtr());

            UserValueType* userValue =
                dynamic_cast<UserValueType*>(data_.user_);

            JSONITY_TYPE_CHECK(userValue != NULL);

            return *userValue;
        }

        template<typename UserValueType>
        const UserValueType& getUserValue() const
        {
            JSONITY_TYPE_CHECK(isUserValue() || isUserValuePtr());

            UserValueType* userValue =
                dynamic_cast<UserValueType*>(data_.user_);

            JSONITY_TYPE_CHECK(userValue != NULL);

            return *userValue;
        }

        template<typename UserValueType>
        UserValueType* getUserValuePtr()
        {
            JSONITY_TYPE_CHECK(isUserValue() || isUserValuePtr());
            return dynamic_cast<UserValueType*>(data_.user_);
        }

        template<typename UserValueType>
        const UserValueType* getUserValue() const
        {
            JSONITY_TYPE_CHECK(isUserValue() || isUserValuePtr());
            return dynamic_cast<UserValueType*>(data_.user_);
        }

    public:

        // Setter

        void setNumber(int32_t number)
        {
            destroy();
            assignNumber(number);
        }

        void setNumber(int64_t number)
        {
            destroy();
            assignNumber(number);
        }

        void setString(const char_t* str)
        {
            destroy();
            assignString(str);
        }

        void setString(const String& str)
        {
            destroy();
            assignString(str.c_str(), str.length());
        }

        void setBoolean(bool boolean)
        {
            destroy();
            assignBoolean(boolean);
        }

        void setReal(double real)
        {
            destroy();
            assignReal(real);
        }

        template<typename KeyType, typename ValueType>
        void setObject(const std::map<KeyType, ValueType>& map)
        {
            destroy();
            assignObject(map);
        }

        void setUserValue(const UserValueBase& userValue)
        {
            destroy();
            assignUserValue(userValue);
        }

        void setUserValue(UserValueBase* userValue)
        {
            destroy();
            assignUserValuePtr(userValue);
        }

        void setValue(const Value& value)
        {
            if (this == &value)
            {
                return;
            }
            destroy();
            assignValue(value);
        }

    public:

        void clear()
        {
            if (isArray())
            {
                getArray().clear();
            }
            else if (isObject())
            {
                getObject().clear();
            }
            else if (isString())
            {
                getString().clear();
            }
            else if (isNumber())
            {
                setNumber(0);
            }
            else if (isBoolean())
            {
                setBoolean(false);
            }
            else if (isReal())
            {
                setReal(0.0);
            }
        }

        bool isEmpty() const
        {
            if (isArray())
            {
                return getArray().empty();
            }
            else if (isObject())
            {
                return getObject().empty();
            }
            else if (isString())
            {
                return getString().empty();
            }
            else if (isNumber())
            {
                return (getNumber() == 0);
            }
            else if (isBoolean())
            {
                return (getBoolean() == false);
            }
            else if (isReal())
            {
                return (getReal() == 0.0);
            }

            return true;
        }

        size_t getSize() const
        {
            if (isArray())
            {
                return getArray().size();
            }
            else if (isObject())
            {
                return getObject().size();
            }
            else if (isString())
            {
                return getString().size();
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

    public:

        // Convert

        int64_t toNumber() const
        {
            if (isString())
            {
                int64_t number = 0;
                stringToInt(getString(), number);
                return number;
            }
            else if (isNumber() || isReal() || isBoolean())
            {
                return getNumber();
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        String toString() const
        {
            if (isNumber())
            {
                OStringStream oss;
                intToStream(getNumber(), oss);
                return oss.str();
            }
            else if (isReal())
            {
                OStringStream oss;
                doubleToStream(getReal(), oss);
                return oss.str();
            }
            else if (isBoolean())
            {
                return String(1, (getBoolean() ?
                    JSONITY_CHAR('1') : JSONITY_CHAR('0')));
            }
            else if (isString())
            {
                return getString();
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

    public:

        // Array operation
        
        Value& operator[](int index)
        {
            JSONITY_TYPE_CHECK(isArray());
            JSONITY_ASSERT(index >= 0);
            JSONITY_ASSERT(getSize() > static_cast<size_t>(index));
        
            return getArray()[static_cast<size_t>(index)];
        }

        Value& addNewValue()
        {
            JSONITY_TYPE_CHECK(isArray());
#ifdef JSONITY_SUPPORT_CXX_11
            getArray().emplace_back(Value());
#else
            getArray().push_back(Value());
#endif
            return getArray().back();
        }

    public:

        // Object operation

        bool hasName(const String& name) const
        {
            JSONITY_TYPE_CHECK(isObject());
            return (getObject().find(name) != getObject().end());
        }

        bool find(const String& name, Value& value) const
        {
            JSONITY_TYPE_CHECK(isObject());

            typename Object::const_iterator it =
                getObject().find(name);
            if (it != getObject().end())
            {
                value = it->second;
                return true;
            }

            return false;
        }

        template<typename ContainerType>
        size_t findRecursive(const String& name,
                             ContainerType& container) const
        {
            JSONITY_TYPE_CHECK(isObject());

            for (typename Object::const_iterator itObj =
                    getObject().begin();
                itObj != getObject().end(); ++itObj)
            {
                if (itObj->first == name)
                {
                    container.push_back(itObj->second);
                }
                else if (itObj->second.isArray())
                {
                    for (typename Array::const_iterator itArr =
                            itObj->second.getArray().begin();
                        itArr != itObj->second.getArray().end(); ++itArr)
                    {
                        if (itArr->isObject())
                        {
                            itArr->findRecursive(name, container);
                        }
                    }
                }
                else if (itObj->second.isObject())
                {
                    itObj->second.findRecursive(name, container);
                }
            }

            return container.size();
        }

        Value& operator[](const char_t* name)
        {
            if (isNull())
            {
                assignObject(Object());
            }

            if (isObject())
            {
                return getObject()[name];
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        Value& operator[](const String& name)
        {
            return operator[](name.c_str());
        }

    public:

        // Compare

        int32_t compare(int32_t number,
                        bool ignoreOrder = true) const
        {
            ((void)ignoreOrder);

            if (isNumber() || isBoolean())
            {
                return static_cast<int32_t>(
                    getNumber() - number);
            }
            else if (isReal())
            {
                return compare((double)number);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        int32_t compare(int64_t number,
                        bool ignoreOrder = true) const
        {
            ((void)ignoreOrder);

            if (isNumber() || isBoolean())
            {
                return static_cast<int32_t>(
                    getNumber() - number);
            }
            else if (isReal())
            {
                return compare((double)number);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        int32_t compare(const char_t* str,
                        bool ignoreOrder = true) const
        {
            JSONITY_TYPE_CHECK(isString());
            ((void)ignoreOrder);
            return getString().compare(str);
        }

        int32_t compare(const String& str,
                        bool ignoreOrder = true) const
        {
            JSONITY_TYPE_CHECK(isString());
            ((void)ignoreOrder);
            return getString().compare(str);
        }

        int32_t compare(bool boolean,
                        bool ignoreOrder = true) const
        {
            ((void)ignoreOrder);

            if (isBoolean() || isNumber() || isReal())
            {
                return (getBoolean() - boolean);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        int32_t compare(double real,
                        bool ignoreOrder = true) const
        {
            ((void)ignoreOrder);

            if (isReal() || isNumber() || isBoolean())
            {
                if (getReal() < real)
                {
                    return -1;
                }
                else if (getReal() > real)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

        template<typename ConatainerType>
        int32_t compare(const ConatainerType& container,
                        bool ignoreOrder = true) const
        {
            JSONITY_TYPE_CHECK(isArray());

            if (getSize() < container.size())
            {
                return -1;
            }
            else if (getSize() > container.size())
            {
                return 1;
            }
   
            typename ConatainerType::const_iterator itOther =
                container.begin();

            if (ignoreOrder)
            {
                Array arr = getArray(); // copy

                for (; itOther != container.end(); ++itOther)
                {
                    typename Array::iterator itArr =
                        arr.begin();

                    for (; itArr != arr.end(); ++itArr)
                    {
                        if (itArr->compare(*itOther, ignoreOrder) == 0)
                        {
                            break;
                        }
                    }

                    if (itArr == arr.end())
                    {
                        return -1;
                    }

                    arr.erase(itArr);
                }
            }
            else
            {
                typename Array::const_iterator itArr =
                    getArray().begin(); 

                for (; itOther != container.end(); ++itOther)
                {
                    int32_t result =
                        itArr->compare(*itOther, ignoreOrder);
                    if (result != 0)
                    {
                        return result;
                    }
                    ++itArr;
                }
            }

            return 0;
        }

        template<typename KeyType, typename ValueType>
        int32_t compare(const std::map<KeyType, ValueType>& map,
                        bool ignoreOrder = true) const
        {
            JSONITY_TYPE_CHECK(isObject());

            if (getSize() < map.size())
            {
                return -static_cast<int32_t>(map.size() - getSize());
            }
            else if (getSize() > map.size())
            {
                return static_cast<int32_t>(getSize() - map.size());
            }

            for (typename std::map<KeyType, ValueType>::
                    const_iterator itOther = map.begin();
                itOther != map.end(); ++itOther)
            {
                Value name(itOther->first);

                typename Object::const_iterator it;

                if (name.isString())
                {
                    it = getObject().find(name.getString());
                }
                else
                {
                    it = getObject().find(name.toString());
                }

                if (it == getObject().end())
                {
                    return -1;
                }

                int32_t result =
                    it->second.compare(itOther->second, ignoreOrder);
                if (result != 0)
                {
                    return result;
                }
            }

            return 0;
        }

        int32_t compare(const Value& value,
                        bool ignoreOrder = true) const
        {
            if (isNull() && value.isNull())
            {
                return 0;
            }
            else if (isNumber() &&
                (value.isNumber() || value.isBoolean()))
            {
                return compare(value.getNumber());
            }
            else if (isNumber() && value.isReal())
            {
                return compare(value.getReal());
            }
            else if (isString() && value.isString())
            {
                return compare(value.getString());
            }
            else if (isBoolean() &&
                (value.isBoolean() || value.isNumber() || value.isReal()))
            {
                return compare(value.getBoolean());
            }
            else if (isReal() &&
                (value.isReal() || value.isNumber() || value.isBoolean()))
            {
                return compare(value.getReal());
            }
            else if (isArray() && value.isArray())
            {
                return compare(value.getArray(), ignoreOrder);
            }
            else if (isObject() && value.isObject())
            {
                return compare(value.getObject(), ignoreOrder);
            }
            else
            {
                JSONITY_THROW_TYPE_MISMATCH();
            }
        }

    public:

        // Assignment operator

        Value& operator=(int32_t number)
        {
            setNumber(number);
            return *this;
        }

        Value& operator=(int64_t number)
        {
            setNumber(number);
            return *this;
        }

        Value& operator=(const char_t* str)
        {
            setString(str);
            return *this;
        }

        Value& operator=(const String& str)
        {
            setString(str);
            return *this;
        }

        Value& operator=(bool boolean)
        {
            setBoolean(boolean);
            return *this;
        }

        Value& operator=(double real)
        {
            setReal(real);
            return *this;
        }

        template<typename KeyType, typename ValueType>
        Value& operator=(const std::map<KeyType, ValueType>& map)
        {
            setObject(map);
            return *this;
        }

        Value& operator=(const UserValueBase& userValue)
        {
            setUserValue(userValue);
            return *this;
        }

        Value& operator=(UserValueBase* userValue)
        {
            setUserValue(userValue);
            return *this;
        }

        Value& operator=(const UserValueBase* userValue)
        {
            setUserPtrValue(userValue);
            return *this;
        }

        Value& operator=(const Value& other)
        {
            setValue(other);
            return *this;
        }

#ifdef JSONITY_SUPPORT_CXX_11
        Value& operator=(Value&& other)
        {
            move(std::move(other));
            return *this;
        }
#endif

    public:

        // Cast operator

        operator int32_t() const
        {
            return static_cast<int32_t>(getNumber());
        }

        operator int64_t() const
        {
            return getNumber();
        }

        operator const char_t*() const
        {
            return getString().c_str();
        }

        operator String&()
        {
            return getString();
        }

        operator bool() const
        {
            return getBoolean();
        }

        operator double() const
        {
            return getReal();
        }

        operator Array&()
        {
            return getArray();
        }

        operator Object&()
        {
            return getArray();
        }

    public:

        bool operator==(int32_t number) const
        {
            return (compare(number) == 0);
        }

        bool operator==(int64_t number) const
        {
            return (compare(number) == 0);
        }

        bool operator==(const char_t* str) const
        {
            return (compare(str) == 0);
        }

        bool operator==(const String& str) const
        {
            return (compare(str) == 0);
        }

        bool operator==(bool boolean) const
        {
            return (compare(boolean) == 0);
        }

        bool operator==(double real) const
        {
            return (compare(real) == 0);
        }

        template<typename ContainerType>
        bool operator==(const ContainerType& container) const
        {
            return (compare(container) == 0);
        }

        template<typename KetType, typename ValueType>
        bool operator==(const std::map<KetType, ValueType>& map) const
        {
            return (compare(map) == 0);
        }

        bool operator==(const Value& value) const
        {
            return (compare(value) == 0);
        }

    public:

        bool operator!=(int32_t number) const
        {
            return !operator==(number);
        }

        bool operator!=(int64_t number) const
        {
            return !operator==(number);
        }

        bool operator!=(const char_t* str) const
        {
            return !operator==(str);
        }

        bool operator!=(const String& str) const
        {
            return !operator==(str);
        }

        bool operator!=(bool boolean) const
        {
            return !operator==(boolean);
        }

        bool operator!=(double real) const
        {
            return !operator==(real);
        }

        template<typename ContainerType>
        bool operator!=(const ContainerType& container) const
        {
            return !operator==(container);
        }

        template<typename KeyName, typename ValueType>
        bool operator!=(const std::map<KeyName, ValueType>& map) const
        {
            return !operator==(map);
        }

        bool operator!=(const Value& value) const
        {
            return !operator==(value);
        }

    public:

        bool operator<(int32_t number) const
        {
            return (compare(number) < 0);
        }

        bool operator<(int64_t number) const
        {
            return (compare(number) < 0);
        }

        bool operator<(const char_t* str) const
        {
            return (compare(str) < 0);
        }

        bool operator<(const String& str) const
        {
            return (compare(str) < 0);
        }

        bool operator<(bool boolean) const
        {
            return (compare(boolean) < 0);
        }

        bool operator<(double real) const
        {
            return (compare(real) < 0);
        }

        bool operator<(const Value& other) const
        {
            return (compare(other) < 0);
        }

    public:

        bool operator>(int32_t number) const
        {
            return (compare(number) > 0);
        }

        bool operator>(int64_t number) const
        {
            return (compare(number) > 0);
        }

        bool operator>(const char_t* str) const
        {
            return (compare(str) > 0);
        }

        bool operator>(const String& str) const
        {
            return (compare(str) > 0);
        }

        bool operator>(bool boolean) const
        {
            return (compare(boolean) > 0);
        }

        bool operator>(double real) const
        {
            return (compare(real) > 0);
        }

        bool operator>(const Value& other) const
        {
            return (compare(other) > 0);
        }

    private:

        const UserValueBase& getUserValueBase() const
        {
            JSONITY_TYPE_CHECK(isUserValue());
            return *data_.user_;
        }

        const UserValueBase* getUserValueBasePtr() const
        {
            JSONITY_TYPE_CHECK(isUserValuePtr());
            return data_.user_;
        }

        void encodeUserValue(EncodeContext& ctx) const
        {
            JSONITY_ASSERT(
                isUserValue() || isUserValuePtr());
            data_.user_->encode(ctx);
        }

    private:

        void assignNull()
        {
            type_ = NullType;
        }

        void assignNumber(int64_t number)
        {
            type_ = NumberType;
            data_.n_ = number;
        }

        void assignString(const char_t* str, size_t length)
        {
            type_ = StringType;
            data_.str_ = new String;

            if (length > 0)
            {
                data_.str_->assign(str, length);
            }
        }

        void assignString(const char_t* str)
        {
            type_ = StringType;
            data_.str_ = new String(str);
        }

        void assignBoolean(bool boolean)
        {
            type_ = BooleanType;
            data_.b_ = boolean;
        }

        void assignReal(double real)
        {
            type_ = RealType;
            data_.d_ = real;
        }

        template<typename ContainerType>
        void assignArray(const ContainerType& container)
        {
            type_ = ArrayType;
            data_.arr_ = new Array;

            std::copy(container.begin(), container.end(),
                std::inserter(getArray(), getArray().end()));
        }

        template<typename KeyType, typename ValueType>
        void assignObject(const std::map<KeyType, ValueType>& map)
        {
            type_ = ObjectType;
            data_.obj_ = new Object;

            for (typename std::map<KeyType, ValueType>::const_iterator it =
                    map.begin();
                it != map.end(); ++it)
            {
                Value name(it->first);

                if (name.isString())
                {
                    getObject()[name.getString()] = it->second;
                }
                else
                {
                    getObject()[name.toString()] = it->second;
                }
            }
        }

        void assignUserValue(const UserValueBase& userValue)
        {
            type_ = UserType;
            data_.user_ = userValue.duplicate();
        }

        void assignUserValuePtr(const UserValueBase* userValue)
        {
            type_ = UserPtrType;
            data_.user_ = const_cast<UserValueBase*>(userValue);
        }

        void assignValue(const Value& value)
        {
            if (value.isNumber())
            {
                assignNumber(value.getNumber());
            }
            else if (value.isString())
            {
                assignString(value.getString().c_str());
            }
            else if (value.isBoolean())
            {
                assignBoolean(value.getBoolean());
            }
            else if (value.isReal())
            {
                assignReal(value.getReal());
            }
            else if (value.isArray())
            {
                assignArray(value.getArray());
            }
            else if (value.isObject())
            {
                assignObject(value.getObject());
            }
            else if (value.isUserValue())
            {
                assignUserValue(value.getUserValueBase());
            }
            else if (value.isUserValuePtr())
            {
                assignUserValuePtr(value.getUserValueBasePtr());
            }
            else
            {
                assignNull();
            }
        }

#ifdef JSONITY_SUPPORT_CXX_11
        void move(Value&& other)
        {
            destroy();
            type_ = other.type_;
            data_ = other.data_;
            other.type_ = NullType;
        }
#endif

    private:
        Type type_;
        union Data {
            int64_t n_;
            String* str_;
            bool b_;
            double d_;
            Array* arr_;
            Object* obj_;
            UserValueBase* user_;
        } data_;

        friend class JsonBase;

    }; // class JsonBasic::Value

public:

    //-----------------------------------------------------------------------//
    // JsonBasic::Cursor
    //-----------------------------------------------------------------------//

    class Cursor
    {
    public:
        Cursor()
        {
            pos_ = 0;
            row_ = 0;
            col_ = 0;
        }

    public:
        uint32_t getPos() const
        {
            return pos_;
        }

        uint32_t getRow() const
        {
            return row_;
        }

        uint32_t getCol() const
        {
            return col_;
        }

    private:
        void nextRow()
        {
            ++row_;
            col_ = 0;
        }

        void nextCol()
        {
            ++col_;
        }

        void setPos(uint32_t pos)
        {
            pos_ = pos;
        }

        uint32_t pos_;
        uint32_t row_;
        uint32_t col_;

        friend class JsonBase;

    }; // class JsonBasic::Cursor

    //-----------------------------------------------------------------------//
    // JsonBasic::Error
    //-----------------------------------------------------------------------//

    class Error
    {
    public:
        Error()
        {
            proc_ = 0;
            errorCode_ = 0;
            codeLine_ = 0;
        }

        // Process
        static const int32_t ObjectProc = 1; 
        static const int32_t ArrayProc = 2; 
        static const int32_t NumberProc = 3; 
        static const int32_t StringProc = 4; 
        static const int32_t TrueProc = 5;
        static const int32_t FalseProc = 6; 
        static const int32_t RealProc = 7;
        static const int32_t NullProc = 8;
        static const int32_t CodePointProc = 9;
        static const int32_t UnknownProc = 10;

        // Error code
        static const int32_t UnexpectedEOF = 101;
        static const int32_t UnexpectedToken = 102;
        static const int32_t UnexpectedComma = 103;
        static const int32_t UnexpectedColon = 104;
        static const int32_t UnexpectedEscapeToken = 105;
        static const int32_t ExpectedLCB = 106;     // '{'
        static const int32_t ExpectedLSB = 107;     // '['
        static const int32_t ExpectedQuot = 108;    // '"'
        static const int32_t ExpectedComma = 109;   // ','
        static const int32_t ExpectedColon = 110;   // ':'
        static const int32_t NameAlreadyExists = 111;
        static const int32_t InvalidHex = 112;
        static const int32_t InvalidSurrogatePair = 113;
        static const int32_t SyntaxError = 114;
        static const int32_t NotSupported = 115;

    public:

        bool isEmpty() const
        {
            return (proc_ == 0);
        }

        const Cursor& getCursor() const
        {
            return cur_;
        }

        int32_t getProcess() const
        {
            return proc_;
        }

        int32_t getErrorCode() const
        {
            return errorCode_;
        }

        int32_t getCodeLine() const
        {
            return codeLine_;
        }

    private:
        void setError(const Cursor& cur, int32_t proc,
                      int32_t errorCode, int32_t codeLine)
        {
            cur_ = cur;
            proc_ = proc;
            errorCode_ = errorCode;
            codeLine_ = codeLine;
        }

        Cursor cur_;
        int32_t proc_;
        int32_t errorCode_;
        int32_t codeLine_;

        friend class JsonBase;

    }; // class JsonBasic::Error

    //-----------------------------------------------------------------------//
    // JsonBasic::Exception
    //-----------------------------------------------------------------------//

    class Exception : public std::exception
    {
    public:
        Exception(int32_t codeLine, const std::string& message)
            : codeLine_(codeLine), message_(message)
        {
        }

        virtual ~Exception() throw()
        {
        }

    public:
        virtual const char* what() const throw()
        {
            return message_.c_str();
        }

        int32_t getCodeLine() const
        {
            return codeLine_;
        }

    private:
        int32_t codeLine_;
        std::string message_;

    }; // clsss JsonBasic::Exception

    class TypeMismatchException : public Exception
    {
    public:
        TypeMismatchException(int32_t codeLine)
            : Exception(codeLine, "Json::TypeMismatchException")
        {
        }

    }; // clsss JsonBasic::TypeMismatchException

    //-----------------------------------------------------------------------//
    // JsonBasic::EncodeStyle
    //-----------------------------------------------------------------------//

    class EncodeStyle
    {
    public:
        EncodeStyle()
        {
            style_ = 0;
            separator_ = JSONITY_CHAR(' ');
            setStandardStyle();
        }

        static const uint32_t IndentStyle = 0x01;
        static const uint32_t NewLineStyle = 0x02;
        static const uint32_t QuatStyle = 0x04;
        static const uint32_t EscapeCtrlCharStyle = 0x08;
        static const uint32_t PrintNewLineStyle = 0x10;
        static const uint32_t CRLFStyle = 0x100;

    public:
        void setStandardStyle()
        {
            setIndent(true);
            setNewLine(true);
            setEscapeCtrlChar(false);
            setQuat(false);
            setPrintNewLine(false);
        }

        void setIndent(bool enable,
                       char_t indentChar = JSONITY_CHAR(' '),
                       size_t indentCharCount = 4)
        {
            if (enable)
            {
                style_ |= IndentStyle;
            }
            else
            {
                style_ &= ~IndentStyle;
            }

            indentChar_ = indentChar;
            indentCharCount_ = indentCharCount;
        }

        void setNewLine(bool enable,
#ifdef JSONITY_OS_WINDOWS
                        bool CRLF = true)
#else
                        bool CRLF = false)
#endif
        {
            if (enable)
            {
                style_ |= NewLineStyle;
            }
            else
            {
                style_ &= ~NewLineStyle;
            }

            if (CRLF)
            {
                style_ |= CRLFStyle;
            }
            else
            {
                style_ &= ~CRLFStyle;
            }
        }

        void setEscapeCtrlChar(bool enable)
        {
            if (enable)
            {
                style_ |= EscapeCtrlCharStyle;
            }
            else
            {
                style_ &= ~EscapeCtrlCharStyle;
            }
        }

        void setQuat(bool enable)
        {
            if (enable)
            {
                style_ |= QuatStyle;
            }
            else
            {
                style_ &= ~QuatStyle;
            }
        }

        void setPrintNewLine(bool enable)
        {
            if (enable)
            {
                style_ |= PrintNewLineStyle;
            }
            else
            {
                style_ &= ~PrintNewLineStyle;
            }
        }

    private:
        char_t getIndentChar() const
        {
            return indentChar_;
        }

        size_t getIndentCharCount() const
        {
            return indentCharCount_;
        }

        char_t getSeparatorChar() const
        {
            return separator_;
        }

        bool isCRLF() const
        {
            return ((style_ & CRLFStyle) == CRLFStyle);
        }

        bool isEnableIndent() const
        {
            return ((style_ & IndentStyle) == IndentStyle);
        }

        bool isEnableNewLine() const
        {
            return ((style_ & NewLineStyle) ==
                NewLineStyle);
        }

        bool isEnableEscape() const
        {
            return ((style_ & EscapeCtrlCharStyle) ==
                EscapeCtrlCharStyle);
        }

        bool isEnableQuat() const
        {
            return ((style_ & QuatStyle) == QuatStyle);
        }

        bool isEnablePrintNewLine() const
        {
            return ((style_ & PrintNewLineStyle) ==
                PrintNewLineStyle);
        }

        char_t separator_;
        char_t indentChar_;
        size_t indentCharCount_;
        uint32_t style_;

        friend class JsonBase;

    }; // JsonBasic::EncodeStyle

public:

    //-----------------------------------------------------------------------//
    // JsonBasic::EncodeContext
    //-----------------------------------------------------------------------//

    class EncodeContext
    {
    public:
        EncodeContext(OStream& os, const EncodeStyle* style = NULL)
            : os_(os)
        {
            indent_ = 0;
            style_ = style;
        }

		virtual ~EncodeContext() {}

    public:

        OStream& getOutputStream()
        {
            return os_;
        }

    protected:

        virtual void increaseIndent()
        {
            ++indent_;
        }

        virtual void decreaseIndent()
        {
            --indent_;
        }

        virtual void writeIndent()
        {
            if (style_->isEnableIndent())
            {
                String indentStr;
                indentStr.assign(
                    style_->getIndentCharCount() * indent_,
                    style_->getIndentChar());

                writeString(indentStr.c_str(), indentStr.size());
            }

            if (style_->isEnableNewLine())
            {
                if (style_->isEnableQuat())
                {
                    writeChar(JSONITY_CHAR('"'));
                }
            }
        }

        virtual void writeNewLine()
        {
            if (style_->isEnablePrintNewLine())
            {
                static const char_t str[] =
                {
                    JSONITY_CHAR('\\'),
                    JSONITY_CHAR('r'),
                    JSONITY_CHAR('\\'),
                    JSONITY_CHAR('n'),
                };

                if (style_->isCRLF())
                {
                    writeString(str, 4);
                }
                else
                {
                    writeString(&str[2], 2);
                }
            }

            if (style_->isEnableNewLine())
            {
                if (style_->isEnableQuat())
                {
                    writeChar(JSONITY_CHAR('"'));
                }

                static const char_t str[] =
                {
                    JSONITY_CHAR('\r'),
                    JSONITY_CHAR('\n'),
                };

                if (style_->isCRLF())
                {
                    writeString(str, 2);
                }
                else
                {
                    writeString(&str[1], 1);
                }
            }
        }

        virtual void writeSeparator()
        {
            writeChar(style_->getSeparatorChar());
        }

        virtual void writeEscape()
        {
            if (style_->isEnableEscape())
            {
                writeChar(JSONITY_CHAR('\\'));
            }
        }

    private:
        void writeString(const char_t* str, size_t size)
        {
            getOutputStream().write(str, size);
        }

        void writeChar(char_t ch)
        {
            writeString(&ch, 1);
        }

    private:
        OStream& os_;
        size_t indent_;
        const EncodeStyle* style_;

        EncodeContext(const EncodeContext&);
        EncodeContext& operator=(const EncodeContext&);

        friend class JsonBase;

    }; // JsonBasic::EncodeContext

    class UnreadableEncodeContext : public EncodeContext
    {
    public:
        UnreadableEncodeContext(OStream& os)
            : EncodeContext(os) {}
    public:
        void increaseIndent() {}
        void decreaseIndent() {}
        void writeIndent() {}
        void writeNewLine() {}
        void writeSeparator() {}
        void writeEscape() {}
    };

public:

    //-----------------------------------------------------------------------//
    // JsonBase public methods
    //-----------------------------------------------------------------------//

    // Decode

    static bool decode(IStream& is, Value& value,
                       Error* error = NULL)
    {
        StreamDecodeContext ctx(is);

        if (!decodeValue(ctx, value))
        {
            if (error != NULL)
            {
                ctx.getError(*error);
            }

            return false;
        }

        return true;
    }

    static bool decode(const char_t* jsonStr, Value& value,
                       Error* error = NULL)
    {
        StringDecodeContext ctx(jsonStr);

        if (!decodeValue(ctx, value))
        {
            if (error != NULL)
            {
                ctx.getError(*error);
            }

            return false;
        }

        return true;
    }

    static bool decode(const String& jsonStr, Value& value,
                       Error* error = NULL)
    {
        return decode(jsonStr.c_str(), value, error);
    }


    // Encode

    static void encode(const Value& value, OStream& os,
                       const EncodeStyle* style = NULL)
    {
        if (style == NULL)
        {
            UnreadableEncodeContext ctx(os);
            encodeValue(ctx, value);
        }
        else
        {
            EncodeContext ctx(os, style);
            ctx.writeIndent();
            encodeValue(ctx, value);
            ctx.writeNewLine();
        }
    }

    static void encode(const Value& value, String& jsonStr,
                       const EncodeStyle* style = NULL)
    {
        OStringStream oss;
        encode(value, oss, style);
        jsonStr = oss.str();
    }


    // Equal

    static bool equal(const Value& value, const char_t* jsonStr,
                      bool ignoreOrder = true, Error* error = NULL)
    {
        Value jsonValue;
     
        if (!decode(jsonStr, jsonValue, error))
        {
            return false;
        }

        return (value.compare(jsonValue, ignoreOrder) == 0);
    }

    static bool equal(const Value& value, const String& jsonStr,
                      bool ignoreOrder = true, Error* error = NULL)
    {
        return equal(value, jsonStr.c_str(), ignoreOrder, error);
    }


    // Null value
    static const Value& null()
    {
        static const Value nullValue;
        return nullValue;
    }

public:

    // Encoder

    static void encodeNull(EncodeContext& ctx)
    {
        static const char_t str[] =
        {
            JSONITY_CHAR('n'),
            JSONITY_CHAR('u'),
            JSONITY_CHAR('l'),
            JSONITY_CHAR('l'),
        };

        ctx.writeString(str, 4);
    }

    static void encodeNumber(EncodeContext& ctx, int64_t number)
    {
        intToStream(number, ctx.getOutputStream());
    }

    static void encodeString(EncodeContext& ctx, const String& str)
    {
        ctx.writeEscape();
        ctx.writeChar(JSONITY_CHAR('\"'));

        const char_t* cur = str.c_str();
        const char_t* head = cur;

        while ((*cur) != JSONITY_CHAR('\0'))
        {
            char_t escapePairCh = JSONITY_CHAR('\0');

            if ((*cur) == JSONITY_CHAR('"'))
            {
                escapePairCh = JSONITY_CHAR('\"');
            }
            else if ((*cur) == JSONITY_CHAR('\\'))
            {
                escapePairCh = JSONITY_CHAR('\\');
            }
            else if ((*cur) == JSONITY_CHAR('/'))
            {
                escapePairCh = JSONITY_CHAR('/');
            }
            else if ((*cur) == JSONITY_CHAR('\b'))
            {
                escapePairCh = JSONITY_CHAR('b');
            }
            else if ((*cur) == JSONITY_CHAR('\f'))
            {
                escapePairCh = JSONITY_CHAR('f');
            }
            else if ((*cur) == JSONITY_CHAR('\n'))
            {
                escapePairCh = JSONITY_CHAR('n');
            }
            else if ((*cur) == JSONITY_CHAR('\r'))
            {
                escapePairCh = JSONITY_CHAR('r');
            }
            else if ((*cur) == JSONITY_CHAR('\t'))
            {
                escapePairCh = JSONITY_CHAR('t');
            }
            else if (((uint32_t)(*cur) < JSONITY_CHAR(0x20)) ||
                     ((uint32_t)(*cur) == JSONITY_CHAR(0x7f)))
            {
                escapePairCh = JSONITY_CHAR('u');
            }

            if (escapePairCh != JSONITY_CHAR('\0'))
            {
                size_t size = static_cast<size_t>(cur - head);
                if (size > 0)
                {
                    ctx.writeString(head, size);
                }

                ctx.writeEscape();

                char_t escapeStr[2];
                escapeStr[0] = JSONITY_CHAR('\\');
                escapeStr[1] = escapePairCh;

                ctx.writeString(escapeStr, 2);

                if (escapePairCh == JSONITY_CHAR('u'))
                {
                    char_t hexStr[4];
                    hexStr[0] = JSONITY_CHAR('0');
                    hexStr[1] = JSONITY_CHAR('0');
                    encodeHex((uint32_t)(((*cur) & 0xf0) >> 4), hexStr[2]);
                    encodeHex((uint32_t)(((*cur) & 0x0f)), hexStr[3]);

                    ctx.writeString(hexStr, 4);
                }

                ++cur;
                head = cur;
            }
            else
            {
                ++cur;
            }
        }

        size_t size = static_cast<size_t>(cur - head);
        if (size > 0)
        {
            ctx.getOutputStream().write(head, size);
        }

        ctx.writeEscape();
        ctx.writeChar(JSONITY_CHAR('\"'));
    }

    static void encodeBoolean(EncodeContext& ctx, bool boolean)
    {
        if (boolean)
        {
            static const char_t str[] =
            {
                JSONITY_CHAR('t'),
                JSONITY_CHAR('r'),
                JSONITY_CHAR('u'),
                JSONITY_CHAR('e')
            };

            ctx.writeString(str, 4);
        }
        else
        {
            static const char_t str[] =
            {
                JSONITY_CHAR('f'),
                JSONITY_CHAR('a'),
                JSONITY_CHAR('l'),
                JSONITY_CHAR('s'),
                JSONITY_CHAR('e')
            };

            ctx.writeString(str, 5);
        }
    }

    static void encodeReal(EncodeContext& ctx, double real)
    {
        doubleToStream(real, ctx.getOutputStream());
    }

    static void encodeArray(EncodeContext& ctx, const Array& arr)
    {
        ctx.writeChar(JSONITY_CHAR('['));

        if (!arr.empty())
        {
            ctx.writeNewLine();
            ctx.increaseIndent();

            for (typename Array::const_iterator it = arr.begin();
                it != arr.end();)
            {
                ctx.writeIndent();

                encodeValue(ctx, *it);
                ++it;

                if (it != arr.end())
                {
                    ctx.writeChar(JSONITY_CHAR(','));
                    ctx.writeNewLine();
                }
            }

            ctx.writeNewLine();
            ctx.decreaseIndent();
            ctx.writeIndent();
        }

        ctx.writeChar(JSONITY_CHAR(']'));
    }

    static void encodeArray(EncodeContext& ctx, const Value& value)
    {
        encodeArray(ctx, value.getArray());
    }

    static void encodeObject(EncodeContext& ctx, const Object& obj)
    {
        ctx.writeChar(JSONITY_CHAR('{'));

        if (!obj.empty())
        {
            ctx.writeNewLine();
            ctx.increaseIndent();

            for (typename Object::const_iterator it = obj.begin();
                it != obj.end();)
            {
                ctx.writeIndent();

                encodeString(ctx, it->first);

                ctx.writeChar(JSONITY_CHAR(':'));
                ctx.writeSeparator();

                encodeValue(ctx, it->second);

                ++it;

                if (it != obj.end())
                {
                    ctx.writeChar(JSONITY_CHAR(','));
                    ctx.writeNewLine();
                }
            }

            ctx.writeNewLine();
            ctx.decreaseIndent();
            ctx.writeIndent();
        }

        ctx.writeChar(JSONITY_CHAR('}'));
    }

    static void encodeObject(EncodeContext& ctx, const Value& value)
    {
        encodeObject(ctx, value.getObject());
    }

    static void encodeValue(EncodeContext& ctx, const Value& value)
    {
        if (value.isNull())
        {
            encodeNull(ctx);
        }
        else if (value.isNumber())
        {
            encodeNumber(ctx, value.getNumber());
        }
        else if (value.isString())
        {
            encodeString(ctx, value.getString());
        }
        else if (value.isBoolean())
        {
            encodeBoolean(ctx, value.getBoolean());
        }
        else if (value.isReal())
        {
            encodeReal(ctx, value.getReal());
        }
        else if (value.isArray())
        {
            encodeArray(ctx, value.getArray());
        }
        else if (value.isObject())
        {
            encodeObject(ctx, value.getObject());
        }
        else if (value.isUserValue())
        {
            value.encodeUserValue(ctx);
        }
        else if (value.isUserValuePtr())
        {
            value.encodeUserValue(ctx);
        }
    }

private:

    class DecodeContext
    {
    public:
        DecodeContext()
        {
            proc_ = 0;
            errorCode_ = 0;
            codeLine_ = 0;
        }

        virtual ~DecodeContext() {}

    public:
        virtual char_t getCurrentChar() const = 0;
        virtual void nextChar() = 0;
        virtual void nextLine() = 0;
        virtual size_t getCurrentPos() const = 0;
        virtual void savePos() = 0;
        virtual size_t readFromSavePos(String& str) const = 0;

    public:
        bool isEOF() const
        {
            return (getCurrentChar() == JSONITY_CHAR('\0'));
        }

        void skipWhiteSpace()
        {
            while (!isEOF())
            {
                if ((getCurrentChar() == JSONITY_CHAR(' ')) ||
                    (getCurrentChar() == JSONITY_CHAR('\r')) ||
                    (getCurrentChar() == JSONITY_CHAR('\t')))
                {
                    nextChar();
                    continue;
                }
                else if (getCurrentChar() == JSONITY_CHAR('\n'))
                {
                    nextLine();
                    continue;
                }
                break;
            }
        }

        void setError(int32_t proc, int32_t errorCode,
                      int32_t codeLine)
        {
            proc_ = proc;
            errorCode_ = errorCode;
            codeLine_ = codeLine;
            cur_.setPos(static_cast<uint32_t>(getCurrentPos()));
        }

        void getError(Error& error) const
        {
            error.setError(cur_, proc_, errorCode_, codeLine_);
        }

    protected:
        Cursor cur_;
        int32_t proc_;
        int32_t errorCode_;
        int32_t codeLine_;
    };

    class StringDecodeContext : public DecodeContext
    {
    public:
        StringDecodeContext(const char_t* str)
            : headAddr_(str), curAddr_(str), saveAddr_(NULL)
        {
        }

    public:
        char_t getCurrentChar() const
        {
            return *curAddr_;
        }

        void nextChar()
        {
            ++curAddr_;
            DecodeContext::cur_.nextCol();
        }

        void nextLine()
        {
            ++curAddr_;
            DecodeContext::cur_.nextRow();
        }

        size_t getCurrentPos() const
        {
            return (curAddr_ - headAddr_);
        }

        void savePos()
        {
            saveAddr_ = curAddr_;
        }

        size_t readFromSavePos(String& str) const
        {
            size_t size = curAddr_ - saveAddr_;
            if (size == 0)
            {
                return 0;
            }
            str.append(saveAddr_, size);
            return size;
        }

    private:
        StringDecodeContext();
        StringDecodeContext& operator=(const StringDecodeContext&);

        const char_t* headAddr_;
        const char_t* curAddr_;
        const char_t* saveAddr_;
    };

    class StreamDecodeContext : public DecodeContext
    {
    public:
        StreamDecodeContext(IStream& is)
            : is_(is)
        {
        }

    public:
        char_t getCurrentChar() const
        {
            return static_cast<char_t>(is_.rdbuf()->sgetc());
        }

        void nextChar()
        {
            is_.rdbuf()->pubseekoff(
                1, std::ios_base::cur, std::ios_base::in);

            DecodeContext::cur_.nextCol();
        }

        void nextLine()
        {
            is_.rdbuf()->pubseekoff(
                1, std::ios_base::cur, std::ios_base::in);

            DecodeContext::cur_.nextRow();
        }

        size_t getCurrentPos() const
        {
            return static_cast<size_t>(is_.rdbuf()->pubseekoff(
                0, std::ios_base::cur, std::ios_base::in));
        }

        void savePos()
        {
            savePos_ = getCurrentPos();
        }

        size_t readFromSavePos(String& str) const
        {
            std::streamsize size = getCurrentPos() - savePos_;
            if (size == 0)
            {
                return 0;
            }
            size_t index = str.size();
            str.resize(static_cast<size_t>(str.size() + size));

            is_.rdbuf()->pubseekpos(
                savePos_, std::ios_base::in);

            std::streamsize readSize =
#if !defined(JSONITY_OS_WINDOWS) || (_MSC_VER >= 1600)
                is_.rdbuf()->sgetn(&str[index], size);
#else
                is_.rdbuf()->_Sgetn_s(&str[index], size, size);
#endif

            is_.rdbuf()->pubseekpos(
                savePos_ + readSize, std::ios_base::in);

            return static_cast<size_t>(readSize);
        }

    private:
        StreamDecodeContext();
        StreamDecodeContext& operator=(const StreamDecodeContext&);

        IStream& is_;
        size_t savePos_;

    }; // class JsonBase::DecodeContext

private:

    static bool decodeNull(DecodeContext& ctx, Value& value)
    {
        static const char_t str[] =
        {
            JSONITY_CHAR('n'),
            JSONITY_CHAR('u'),
            JSONITY_CHAR('l'),
            JSONITY_CHAR('l'),
            JSONITY_CHAR('\0')
        };

        JSONITY_ASSERT(
            ctx.getCurrentChar() == str[0]);
        ctx.nextChar();

        for (size_t index = 1; index < 4; ++index)
        {
            if (ctx.isEOF())
            {
                ctx.setError(
                    Error::NullProc, Error::UnexpectedEOF,
                    __LINE__);
                return false;
            }

            if (ctx.getCurrentChar() != str[index])
            {
                ctx.setError(
                    Error::NullProc, Error::UnexpectedToken,
                    __LINE__);
                return false;
            }

            ctx.nextChar();
        }

        value.assignNull();

        return true;
    }

    static bool decodeNumber(DecodeContext& ctx, Value& value)
    {
        JSONITY_ASSERT(
             isDigit(ctx.getCurrentChar()) ||
             (ctx.getCurrentChar() == JSONITY_CHAR('-')));

        ctx.savePos();

        if (ctx.getCurrentChar() == JSONITY_CHAR('-'))
        {
            ctx.nextChar();
        }

        bool real = false;

        while (!ctx.isEOF())
        {
            char_t ch = ctx.getCurrentChar();

            if (isDigit(ch))
            {
                ctx.nextChar();
            }
            else if ((ch == JSONITY_CHAR('e')) ||
                     (ch == JSONITY_CHAR('E')) ||
                     (ch == JSONITY_CHAR('-')) ||
                     (ch == JSONITY_CHAR('+')) ||
                     (ch == JSONITY_CHAR('.')))
            {
                real = true;
                ctx.nextChar();
            }
            else
            {
                break;
            }
        }

        String str;
        if (ctx.readFromSavePos(str) == 0)
        {
            ctx.setError(
                Error::NumberProc, Error::UnexpectedToken,
                __LINE__);
            return false;
        }

        if (real)
        {
            double d = 0;
            if (!stringToDouble(str, d))
            {
                ctx.setError(
                    Error::RealProc, Error::SyntaxError,
                    __LINE__);
                return false;
            }
            value.assignReal(d);
        }
        else
        {
            int64_t n = 0;
            if (!stringToInt(str, n))
            {
                ctx.setError(
                    Error::NumberProc, Error::SyntaxError,
                    __LINE__);
                return false;
            }
            value.assignNumber(n);
        }

        return true;
    }

    static bool decodeHex4(DecodeContext& ctx, uint32_t& value)
    {
        value = 0;

        for (size_t count = 0; count < 4; ++count)
        {
            if (ctx.isEOF())
            {
                return false;
            }

            uint32_t hexValue;

            if (!decodeHex(ctx.getCurrentChar(), hexValue))
            {
                return false;
            }

            value <<= 4;
            value += hexValue;

            ctx.nextChar();
        }

        return true;
    }

    static bool decodeCodePoint(DecodeContext& ctx, Value& value)
    {
        JSONITY_ASSERT(value.isString());
        JSONITY_ASSERT(ctx.getCurrentChar() == JSONITY_CHAR('u'));
        ctx.nextChar();

        uint32_t codePoint1;

        if (!decodeHex4(ctx, codePoint1))
        {
            ctx.setError(
                Error::CodePointProc, Error::InvalidHex,
                __LINE__);
            return false;
        }

        if ((codePoint1 >= 0xd800) && (codePoint1 <= 0xdbff))
        {
            if (ctx.getCurrentChar() == JSONITY_CHAR('\\'))
            {
                ctx.nextChar();

                if (ctx.isEOF())
                {
                    ctx.setError(
                        Error::CodePointProc, Error::UnexpectedEOF,
                        __LINE__);
                    return false;
                }

                if (ctx.getCurrentChar() != JSONITY_CHAR('u'))
                {
                    ctx.setError(
                        Error::CodePointProc, Error::UnexpectedToken,
                        __LINE__);
                    return false;
                }

                ctx.nextChar();

                uint32_t codePoint2;

                if (!decodeHex4(ctx, codePoint2))
                {
                    ctx.setError(
                        Error::CodePointProc, Error::InvalidHex,
                        __LINE__);
                    return false;
                }

                if ((codePoint2 < 0xdc00) || (codePoint2 > 0xdfff))
                {
                    ctx.setError(
                        Error::CodePointProc, Error::InvalidSurrogatePair,
                        __LINE__);
                    return false;
                }

                codePoint1 =
                    (((codePoint1 - 0xd800) << 10) |
                     ((codePoint2 - 0xdc00) & 0x3ff))
                    + 0x10000;
            }
        }

        char_t str[4];
        size_t size = 0;

        size_t chSize = sizeof(char_t);     // TODO

        if (chSize == 1)
        {
            if (codePoint1 <= 0x7f)
            {
                str[size++] =
                    static_cast<char_t>(codePoint1 & 0xff);
            }
            else if (codePoint1 <= 0x7ff)
            {
                str[size++] =
                    static_cast<char_t>(0xc0 | ((codePoint1 >> 6) & 0xff));
                str[size++] =
                    static_cast<char_t>(0x80 | ((codePoint1 & 0x3f)));
            }
            else if (codePoint1 <= 0xffff)
            {
                str[size++] =
                    static_cast<char_t>(0xe0 | ((codePoint1 >> 12) & 0xff));
                str[size++] =
                    static_cast<char_t>(0x80 | ((codePoint1 >> 6) & 0x3f));
                str[size++] =
                    static_cast<char_t>(0x80 | (codePoint1 & 0x3f));
            }
            else if (codePoint1 <= 0x10ffff)
            {
                str[size++] =
                    static_cast<char_t>(0xf0 | ((codePoint1 >> 18) & 0xff));
                str[size++] =
                    static_cast<char_t>(0x80 | ((codePoint1 >> 12) & 0x3f));
                str[size++] =
                    static_cast<char_t>(0x80 | ((codePoint1 >> 6) & 0x3f));
                str[size++] =
                    static_cast<char_t>(0x80 | (codePoint1 & 0x3f));
            }
            else
            {
                ctx.setError(
                    Error::CodePointProc, Error::InvalidSurrogatePair,
                    __LINE__);
                return false;
            }
        }
        else if (chSize == 2)
        {
            if (codePoint1 <= 0xffff)
            {
                str[size++] =
                    static_cast<char_t>(codePoint1);
            }
            else if (codePoint1 <= 0x10ffff)
            {
                str[size++] =
                    static_cast<char_t>(
                        0xd800 + ((codePoint1 - 0x10000) >> 10));
                str[size++] =
                    static_cast<char_t>(
                        0xdc00 + ((codePoint1 - 0x10000) & 0x3ff));
            }
            else
            {
                ctx.setError(
                    Error::CodePointProc, Error::InvalidSurrogatePair,
                    __LINE__);
                return false;
            }
        }
        else if (chSize >= 4)
        {
            if (codePoint1 <= 0x10ffff)
            {
                str[size++] =
                    static_cast<char_t>(codePoint1);
            }
            else
            {
                ctx.setError(
                    Error::CodePointProc, Error::InvalidSurrogatePair,
                    __LINE__);
                return false;
            }
        }
        else
        {
            ctx.setError(
                Error::CodePointProc, Error::NotSupported,
                __LINE__);
            return false;
        }

        value.getString().append(str, size);

        return true;
    }

    static bool decodeEscapeChar(DecodeContext& ctx, Value& value)
    {
        JSONITY_ASSERT(value.isString());

        char_t ch = ctx.getCurrentChar();

        if (ch == JSONITY_CHAR('"'))
        {
            value.getString().push_back(JSONITY_CHAR('\"'));
        }
        else if (ch == JSONITY_CHAR('\\'))
        {
            value.getString().push_back(JSONITY_CHAR('\\'));
        }
        else if (ch == JSONITY_CHAR('/'))
        {
            value.getString().push_back(JSONITY_CHAR('/'));
        }
        else if (ch == JSONITY_CHAR('b'))
        {
            value.getString().push_back(JSONITY_CHAR('\b'));
        }
        else if (ch == JSONITY_CHAR('f'))
        {
            value.getString().push_back(JSONITY_CHAR('\f'));
        }
        else if (ch == JSONITY_CHAR('n'))
        {
            value.getString().push_back(JSONITY_CHAR('\n'));
        }
        else if (ch == JSONITY_CHAR('r'))
        {
            value.getString().push_back(JSONITY_CHAR('\r'));
        }
        else if (ch == JSONITY_CHAR('t'))
        {
            value.getString().push_back(JSONITY_CHAR('\t'));
        }
        else if (ch == JSONITY_CHAR('u'))
        {
            return decodeCodePoint(ctx, value);
        }
        else
        {
            ctx.setError(
                Error::StringProc, Error::UnexpectedEscapeToken,
                __LINE__);
            return false;
        }

        ctx.nextChar();

        return true;
    }

    static bool decodeString(DecodeContext& ctx, Value& value)
    {
        JSONITY_ASSERT(
            ctx.getCurrentChar() == JSONITY_CHAR('"'));
        ctx.nextChar();

        value = String();
        bool escape = false;
        ctx.savePos();

        for (;;)
        {
            char_t ch = ctx.getCurrentChar();

            if (!escape && (ch == JSONITY_CHAR('"')))
            {
                break;
            }

            if (ctx.isEOF())
            {
                ctx.setError(
                    Error::StringProc, Error::UnexpectedEOF,
                    __LINE__);
                return false;
            }

            if (!escape)
            {
                if (ch == JSONITY_CHAR('\\'))
                {
                    ctx.readFromSavePos(value.getString());
                    escape = true;
                }

                ctx.nextChar();
            }
            else
            {
                if (!decodeEscapeChar(ctx, value))
                {
                    return false;
                }

                escape = false;
                ctx.savePos();
            }
        }

        ctx.readFromSavePos(value.getString());
        ctx.nextChar();

        return true;
    }

    static bool decodeTrue(DecodeContext& ctx, Value& value)
    {
        static const char_t str[] =
        {
            JSONITY_CHAR('t'),
            JSONITY_CHAR('r'),
            JSONITY_CHAR('u'),
            JSONITY_CHAR('e'),
            JSONITY_CHAR('\0')
        };

        JSONITY_ASSERT(
            ctx.getCurrentChar() == str[0]);
        ctx.nextChar();

        for (size_t index = 1; index < 4; ++index)
        {
            if (ctx.isEOF())
            {
                ctx.setError(
                    Error::TrueProc, Error::UnexpectedEOF,
                    __LINE__);
                return false;
            }

            if (ctx.getCurrentChar() != str[index])
            {
                ctx.setError(
                    Error::TrueProc, Error::UnexpectedToken,
                    __LINE__);
                return false;
            }

            ctx.nextChar();
        }

        value.assignBoolean(true);

        return true;
    }

    static bool decodeFalse(DecodeContext& ctx, Value& value)
    {
        static const char_t str[] =
        {
            JSONITY_CHAR('f'),
            JSONITY_CHAR('a'),
            JSONITY_CHAR('l'),
            JSONITY_CHAR('s'),
            JSONITY_CHAR('e'),
            JSONITY_CHAR('\0')
        };

        JSONITY_ASSERT(
            ctx.getCurrentChar() == str[0]);
        ctx.nextChar();

        for (size_t index = 1; index < 5; ++index)
        {
            if (ctx.isEOF())
            {
                ctx.setError(
                    Error::FalseProc, Error::UnexpectedEOF,
                    __LINE__);
                return false;
            }

            if (ctx.getCurrentChar() != str[index])
            {
                ctx.setError(
                    Error::FalseProc, Error::UnexpectedToken,
                    __LINE__);
                return false;
            }

            ctx.nextChar();
        }

        value.assignBoolean(false);

        return true;
    }

    static bool decodeArray(DecodeContext& ctx, Value& value)
    {
        JSONITY_ASSERT(
            ctx.getCurrentChar() == JSONITY_CHAR('['));
        ctx.nextChar();
        ctx.skipWhiteSpace();

        value.assignArray(Array());

        bool separator = true;

        while (ctx.getCurrentChar() != JSONITY_CHAR(']'))
        {
            if (ctx.isEOF())
            {
                ctx.setError(
                    Error::ArrayProc, Error::UnexpectedEOF,
                    __LINE__);
                return false;
            }

            if (!separator)
            {
                ctx.setError(
                    Error::ArrayProc, Error::ExpectedComma,
                    __LINE__);
                return false;
            }

            if (!decodeValue(ctx, value.addNewValue()))
            {
                value.getArray().pop_back();
                return false;
            }

            ctx.skipWhiteSpace();

            separator = false;
            if (ctx.getCurrentChar() == JSONITY_CHAR(','))
            {
                ctx.nextChar();
                separator = true;
            }
        }

        if (separator && !value.isEmpty())
        {
            ctx.setError(
                Error::ArrayProc, Error::UnexpectedComma,
                __LINE__);
            return false;
        }

        ctx.nextChar();

        return true;
    }

    static bool decodeObject(DecodeContext& ctx, Value& value)
    {
        JSONITY_ASSERT(
            ctx.getCurrentChar() == JSONITY_CHAR('{'));
        ctx.nextChar();
        ctx.skipWhiteSpace();

        value.assignObject(Object());
        
        bool separator = true;

        while (ctx.getCurrentChar() != JSONITY_CHAR('}'))
        {
            if (ctx.isEOF())
            {
                ctx.setError(
                    Error::ObjectProc, Error::UnexpectedEOF,
                    __LINE__);
                return false;
            }

            if (!separator)
            {
                ctx.setError(
                    Error::ObjectProc, Error::ExpectedComma,
                    __LINE__);
                return false;
            }

            Value name;
            if (!decodeString(ctx, name))
            {
                return false;
            }

            if (value.hasName(name))
            {
                ctx.setError(
                    Error::ObjectProc, Error::NameAlreadyExists,
                    __LINE__);
                return false;
            }

            ctx.skipWhiteSpace();

            if (ctx.getCurrentChar() != JSONITY_CHAR(':'))
            {
                ctx.setError(
                    Error::ObjectProc, Error::ExpectedColon,
                    __LINE__);
                return false;
            }

            ctx.nextChar();
            ctx.skipWhiteSpace();

            if (!decodeValue(ctx, value[name.getString()]))
            {
                value.getObject().erase(name);
                return false;
            }
            ctx.skipWhiteSpace();

            separator = false;
            if (ctx.getCurrentChar() == JSONITY_CHAR(','))
            {
                ctx.nextChar();
                ctx.skipWhiteSpace();
                separator = true;
            }
        }

        if (separator && !value.isEmpty())
        {
            ctx.setError(
                Error::ObjectProc, Error::UnexpectedComma,
                __LINE__);
            return false;
        }

        ctx.nextChar();

        return true;
    }

    static bool decodeValue(DecodeContext& ctx, Value& value)
    {
        ctx.skipWhiteSpace();

        if (ctx.isEOF())
        {
            ctx.setError(
                Error::UnknownProc, Error::UnexpectedEOF,
                __LINE__);
            return false;
        }
        else
        {
            char_t ch = ctx.getCurrentChar();

            if (ch == JSONITY_CHAR('"'))
            {
                return decodeString(ctx, value);
            }
            else if (isDigit(ch) ||
                     (ctx.getCurrentChar() == JSONITY_CHAR('-')))
            {
                return decodeNumber(ctx, value);
            }
            else if (ch == JSONITY_CHAR('{'))
            {
                return decodeObject(ctx, value);
            }
            else if (ch == JSONITY_CHAR('['))
            {
                return decodeArray(ctx, value);
            }
            else if (ch == JSONITY_CHAR('t'))
            {
                return decodeTrue(ctx, value);
            }
            else if (ch == JSONITY_CHAR('f'))
            {
                return decodeFalse(ctx, value);
            }
            else if (ch == JSONITY_CHAR('n'))
            {
                return decodeNull(ctx, value);
            }
            else
            {
                ctx.setError(
                    Error::UnknownProc, Error::UnexpectedToken,
                    __LINE__);
                return false;
            }
        }
    }

private:

    static bool isDigit(char_t ch)
    {
        return ((((unsigned)ch) >= JSONITY_CHAR('0')) &&
                (((unsigned)ch) <= JSONITY_CHAR('9')));
    }

    static bool decodeHex(char_t ch, uint32_t& value)
    {
        if (isDigit(ch))
        {
            value = static_cast<uint32_t>(
                ch - JSONITY_CHAR('0'));
        }
        else if ((ch >= JSONITY_CHAR('a')) &&
                 (ch <= JSONITY_CHAR('f')))
        {
            value = static_cast<uint32_t>(
                ch - JSONITY_CHAR('a') + 10);
        }
        else if ((ch >= JSONITY_CHAR('A')) &&
                 (ch <= JSONITY_CHAR('F')))
        {
            value = static_cast<uint32_t>(
                ch - JSONITY_CHAR('A') + 10);
        }
        else
        {
            return false;
        }

        return true;
    }

    static bool encodeHex(uint32_t value, char_t& ch)
    {
        static const char hex[] = "0123456789ABCDEF";

        if (value < 16)
        {
            ch = static_cast<char_t>(hex[value]);
            return true;
        }
        else
        {
            return false;
        }
    }

    static bool stringToInt(const String& src, int64_t& dest)
    {
        bool sign = false;
        const char_t* endp = src.c_str();

        if ((src.length() >= 2) &&
            ((*endp) == JSONITY_CHAR('-')))
        {
            ++endp;
            sign = true;
        }
        else if (src.length() == 0)
        {
            return false;
        }

        int32_t power = 1;

        for (const char_t* pch =
            src.c_str() + (src.length() - 1);
            pch >= endp; --pch)
        {
            char_t ch = *pch;
            if (!isDigit(ch))
            {
                return false;
            }
            dest += ((ch - JSONITY_CHAR('0')) * power);
            power *= 10;
        }

        if (sign)
        {
            dest = -dest;
        }

        return true;
    }

    static bool stringToDouble(const String& src, double& dest)
    {
        bool sign = false;
        const char_t* pch = src.c_str();

        if ((*pch) == JSONITY_CHAR('-'))
        {
            sign = true;
            ++pch;
        }

        int32_t pt = -1;
        int32_t msize = 0;
        for (;;)
        {
            char_t ch = *pch;
            if (!isDigit(ch))
            {
                if ((ch != JSONITY_CHAR('.')) || (pt >= 0))
                {
                    break;
                }
                pt = msize;
            }
            ++pch;
            ++msize;
        }

        const char_t* pexp = pch;
        pch -= msize;
        if (pt < 0)
        {
            pt = msize;
        }
        else
        {
            --msize;
        }

        int32_t fexp = 0;
        if (msize > 18)
        {
            fexp = pt - 18;
            msize = 18;
        }
        else
        {
            fexp = pt - msize;
        }

        if (msize == 0)
        {
            dest = 0.0;
            return true;
        }

        int32_t frac1 = 0;
        for (; msize > 9; --msize)
        {
            char_t ch = *pch;
            ++pch;
            if (ch == '.')
            {
                ch = *pch;
                ++pch;
            }
            frac1 = 10 * frac1 + (ch - JSONITY_CHAR('0'));
        }
        int32_t frac2 = 0;
        for (; msize > 0; --msize)
        {
            char_t ch = *pch;
            ++pch;
            if (ch == JSONITY_CHAR('.'))
            {
                ch = *pch;
                ++pch;
            }
            frac2 = 10 * frac2 + (ch - JSONITY_CHAR('0'));
        }
        dest = (1.0e9 * frac1) + frac2;

        int32_t exp = 0;
        bool esign = false;
        pch = pexp;
        if (((*pch) == JSONITY_CHAR('e')) ||
            ((*pch) == JSONITY_CHAR('E')))
        {
            ++pch;
            if ((*pch) == JSONITY_CHAR('-'))
            {
                esign = true;
                ++pch;
            }
            else if ((*pch) == JSONITY_CHAR('+'))
            {
                ++pch;
            }

            if (!isDigit(*pch))
            {
                return false;
            }

            while (isDigit(*pch))
            {
                exp = exp * 10 +
                    ((*pch) - JSONITY_CHAR('0'));
                ++pch;
            }
        }

        if (esign)
        {
            exp = fexp - exp;
        }
        else
        {
            exp = fexp + exp;
        }

        if (exp < 0)
        {
            esign = true;
            exp = -exp;
        }
        else
        {
            esign = false;
        }

        if (exp >= 512)
        {
            exp = 511;
        }

        double dexp = 1.0;

        static const double powers[] =
        {
            10.0, 100.0, 1.0e4, 1.0e8, 1.0e16,
            1.0e32, 1.0e64, 1.0e128, 1.0e256
        };

        for (const double* power = powers;
            exp != 0; exp >>= 1, ++power)
        {
            if (exp & 01)
            {
                dexp *= *power;
            }
        }

        if (esign)
        {
            dest /= dexp;
        }
        else
        {
            dest *= dexp;
        }

        if (sign)
        {
            dest = -dest;
        }

        return true;
    }

    static void asciiToBasicChar(
        const char* src, char_t* dest, size_t size)
    {
        for (size_t counter = 0; counter < size; ++counter)
        {
            (*dest) = static_cast<char_t>(*src);
            ++src;
            ++dest;
        }
    }

    static void intToStream(int64_t number, OStream& os)
    {
        char str1[256];
        int size = JSONITY_SNPRINTF(
            str1, sizeof(str1), JSONITY_FORM_I64, number);

        size_t chSize = sizeof(char_t); // TODO
        if (chSize == sizeof(char))
        {
            os.write((char_t*)str1, size);
        }
        else
        {
            char_t str2[256];
            asciiToBasicChar(str1, str2, size);

            os.write(str2, size);
        }
    }

    static void doubleToStream(double d, OStream& os)
    {
        char str1[256];
        int size = JSONITY_SNPRINTF(
            str1, sizeof(str1), "%g", d);

        size_t chSize = sizeof(char_t); // TODO
        if (chSize == sizeof(char))
        {
            os.write((char_t*)str1, size);
        }
        else
        {
            char_t str2[256];
            asciiToBasicChar(str1, str2, size);

            os.write(str2, size);
        }
    }

private:
    JsonBase() {}

}; // class JsonBase

//---------------------------------------------------------------------------//
// Json
//---------------------------------------------------------------------------//

typedef JsonBase<char>      u8Json;
typedef JsonBase<char16_t>  u16Json;
typedef JsonBase<char32_t>  u32Json;
typedef JsonBase<wchar_t>   wJson;

JSONITY_VALUE_OPERATOR_IOSTREAM(u8Json)
JSONITY_VALUE_OPERATOR_IOSTREAM(u16Json)
JSONITY_VALUE_OPERATOR_IOSTREAM(u32Json)
JSONITY_VALUE_OPERATOR_IOSTREAM(wJson)

// standard
typedef u8Json  Json;

} // namespace jsonity

#endif//JSONITY_HPP_
