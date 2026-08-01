---
tags:
  - 编程语言
---
# JavaScript
一种前端编程语言，用于处理网页上的各种用户行为，如按钮点击、表单提交、页面加载等, 也可以通过 Node.js  在服务器端运行

## 基础语法

### 变量

变量是用来存储数据的容器。可以使用三种关键字 来声明变量：`var`、`let` 和 `const`

- **`var`**: 在 ES5 及之前的版本中使用，声明的变量可以在整个函数范围内访问，存在变量提升的现象。
- **`let`**: ES6 引入的变量声明方式，具有块级作用域，不会发生变量提升。
- **`const`**: 用于声明常量，声明后无法重新赋值，同样具有块级作用域。

```js
var name = "Alice"; // 使用 var 声明变量
let age = 25; // 使用 let 声明变量
const birthYear = 1998; // 使用 const 声明常量
```

JavaScript 中的基本数据类型包括：
+ 字符串 (String): 用于表示文本数据。
+ 数字 (Number): 用于表示整数和浮动小数。
+ 布尔值 (Boolean): 用于表示 true 或 false。
+ undefined: 变量声明但未赋值时的默认值。
+ null: 表示“无”或“空”值。
+ 对象 (Object): 用于存储多个值的容器。
+ 数组 (Array): 特殊类型的对象，用于存储有序的数据。

```js
let message = "Hello, World!";  // 字符串
let num = 42;                   // 数字
let isActive = true;            // 布尔值
let person = { name: "Alice", age: 25 };  // 对象
let numbers = [1, 2, 3, 4];    // 数组
```

### 运算符

常见的算术运算符包括 `+`、`-`、`*`、`/`、`%`(取余)、`++`(自增)、`--`(自减)
比较运算符用于比较两个值。常见的比较运算符包括 `==`、`===`、`!=`、`!==`、`>`、`<`、`>=`、`<=`
- `==`（相等）: 比较两个值是否相等，忽略数据类型。
- `===`（严格相等）: 比较两个值是否相等，且数据类型也必须相同。
```js
console.log(5 == '5'); // 输出 true (忽略数据类型)
console.log(5 === '5'); // 输出 false (数据类型不同)
```

常见的逻辑运算符有 `&&`（与）、`||`（或）和 `!`（非）
### 流控制

 `if`、`else`、`for`、`while` 等
```js
for (let i = 0; i < 5; i++) {
    console.log(i);  // 输出 0, 1, 2, 3, 4
}
 
let j = 0;
while (j < 5) {
    console.log(j);  // 输出 0, 1, 2, 3, 4
    j++;
}
```
### 函数

函数可以通过关键字 `function` 来声明
```js
function greet(name) {
	console.log("Hello, " + name + "!");
}

greet("Alice"); // 输出 "Hello, Alice!"
```
使用 `return` 语句来指定返回的结果。如果没有显式的返回值，函数默认返回 `undefined`

#### 匿名函数

```js
let sum = function(a, b) {
	return a + b;  
};
console.log(sum(2, 3)); // 输出 5
```
#### 箭头函数

```js
const multiply = (a, b) => a * b;
console.log(multiply(4, 5)); // 输出 20
```
函数体可以省略大括号 `{}` 和 `return`

### 对象

**对象** 是一种复杂数据类型，用于存储一组数据和功能（属性和方法）。对象通常由多个键值对（属性）组成，每个键是一个字符串，值可以是任何数据类型

```js
let person = {
    name: "Alice",
    age: 25,
    greet: function() {
        console.log("Hello, " + this.name);
    }
};
 
console.log(person.name);  // 输出 "Alice"
person.greet();            // 输出 "Hello, Alice"
```

### 数组
使用方括号 `[]` 来创建数组, 数组中的元素用逗号分隔, 下标从 0 开始

- **`push()`**: 向数组末尾添加一个或多个元素。
- **`pop()`**: 移除数组末尾的元素。
- **`shift()`**: 移除数组开头的元素。
- **`unshift()`**: 向数组开头添加一个或多个元素

```js
let numbers = [1, 2, 3];
 
// 添加元素
numbers.push(4);
console.log(numbers);  // 输出 [1, 2, 3, 4]
 
// 移除元素
numbers.pop();
console.log(numbers);  // 输出 [1, 2, 3]
 
// 移除开头元素
numbers.shift();
console.log(numbers);  // 输出 [2, 3]
```

遍历

```js

let colors = ["red", "green", "blue"];
 
// 使用 for 循环遍历
for (let i = 0; i < colors.length; i++) {
    console.log(colors[i]);
}
 
// 使用 forEach 方法遍历
colors.forEach(function(color) {
    console.log(color);
});
```
### 异步

ES6 引入了 **Promise**，它提供了更清晰的异步编程方式。Promise 是一个代表异步操作最终完成（或失败）及其结果值的对象。
一个 Promise 对象有三种状态：`pending`（等待中）、`fulfilled`（已完成）和 `rejected`（已拒绝）。你可以通过 `new Promise()` 创建一个 Promise 对象，并通过 `resolve` 和 `reject` 方法来改变其状态

```c
let myPromise = new Promise((resolve, reject) => {
    let success = true;
    if (success) {
        resolve("操作成功");
    } else {
        reject("操作失败");
    }
});
 
myPromise.then((message) => {
    console.log(message);  // 输出 "操作成功"
}).catch((message) => {
    console.log(message);  // 输出 "操作失败"
});
```

`Promise.all` 方法可以将多个 Promise 对象合并成一个 Promise 对象，它会在所有的 Promise 对象都完成时返回结果。如果其中任何一个 Promise 失败，`Promise.all` 会立刻失败并返回错误

```js

let promise1 = new Promise((resolve) => resolve("任务1完成"));
let promise2 = new Promise((resolve) => resolve("任务2完成"));
let promise3 = new Promise((resolve) => resolve("任务3完成"));
 
Promise.all([promise1, promise2, promise3]).then((results) => {
    console.log(results);  // 输出 ["任务1完成", "任务2完成", "任务3完成"]
});
```

为了进一步简化异步编程，ES7 引入了 `async` 和 `await`。`async` 和 `await` 基于 Promise，但它们使得异步代码看起来更像是同步代码，从而提高了可读性

`async` 关键字用于声明一个异步函数。异步函数总是返回一个 Promise 对象，并且可以使用 `await` 来等待其他异步操作的结果

