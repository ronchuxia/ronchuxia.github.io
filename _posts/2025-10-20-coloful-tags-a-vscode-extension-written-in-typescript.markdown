---
layout: post
title:  "Colorful Tags: A VSCode Extension written in TypeScript"
date:   2025-10-20 23:00:00 -0400
categories: Web
---

I wrote a VSCode extension called "Colorful Tags" that allows users to tag FILES and FOLDERS with colorful labels. The extension is available on the [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=XiaChu.colorful-tags) and the source code is on [Github](https://github.com/ronchuxia/VSCode-Colorful-Tags). More demonstrations can be found on both sites.

![]( {{ '/assets/2025-10-20/ColorfulTags.jpg' | relative_url }} )

# Features

The extensions allows user to:
- Tag files and folders with colorful labels
- Customize tag aliases
- Automatic tag update for moved or renamed files / folders
- Comprehensive right-click context menu that matches Explorer functionality

I wrote this extension so that I can quickly locate important files and folders in a large project.

# Implementation

The extension is implemented using TypeScript and the VSCode Extension API.

## JavaScript, TypeScript, and Node.js

JavaScript is a programming language originally created for **FRONTEND** development to make web pages interactive inside a web browser.

TypeScript is a superset of JavaScript that adds static typing and other features to the language. Static typing enforces types at **COMPILE TIME**. TypeScript code is transpiled to JavaScript before execution. 

JavaScript was meant to run in a web browser. However, Node.js, released in 2009, changed that by creating a server-side runtime for JavaScript. It uses the Chrome V8 engine plus system libraries to let JavaScript:
- Read/write files
- Access the network
- Run servers
- Connect to databases
This means we can now write both the **FRONTEND** and **BACKEND** in the same language - JavaScript.

## TypeScript and VSCode API

As we mentioned earlier, TypeScript is a static typed language. It only enforces types at compile time, and falls back to JavaScript dynamic typing at runtime. For some VSCode APIs, we won't be able to get type information at compile time. In this case, we use TypeScript's `any` type to bypass type checking. This means that we lose type safety for these APIs, which means we need to be extra careful.

For example, the API `commands.registerCommand`:

```typescript
registerCommand(command: string, callback: (args: any[]) => any, thisArg?: any): Disposable
```

Does not check the actual type of `args` passed to the `callback` function.

## Async, Await, and Promise

In JavaScript / TypeScript, we use async / await syntax to handle asynchronous operations (e.g. file operations). 

### Async

Async does two main things:
1. It wraps the entire function in a Promise.
    When you write:
    ```JavaScript
    async function load(): Promise<void> {
    console.log("Start");
    console.log("End");
    }
    ```

    JavaScript transforms it into something like this behind the scenes:
    ```JavaScript
    function load(): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log("Start");
        console.log("End");
        resolve(); // signals "I'm done!"
    });
    }
    ```
2. It allows you to use await inside the function.

### Await

When JavaScript hits the await line:
1. It pauses execution of the activate function at that line.
2. It lets other code run.
3. When the awaited operation finishes, it resumes from that exact line.
4. Then continues to the next line.

Async / await is like interrupts: os switches cpu to other threads at await, when await finishes execution, an interrupt switches cpu back to await line.

### Promise

A Promise represents a value that will be available in the future — something that’s still “pending” but will eventually either succeed (resolve) or fail (reject).

VSCode API uses the interface `Thenable<T>` instead of `Promise<T>` in many type definitions. A Thenable is anything that has a `.then()` method. It allows interoperability between native and custom Promise implementations.

### Example

In the extension, I use `globalState` to store tags across sessions.
- `globalState.update` is an async function. It writes data to disk, which may take some time.
- `globalState.get` is a synchronous function. It reads data from memory, which is fast.

