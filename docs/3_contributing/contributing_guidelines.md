# How to contribute

* [Introduction](#introduction)
* [Programming guidelines](#programming-guidelines)
* [Fixing a bug](#fixing-a-bug)
* [Adding a new feature](#adding-a-new-feature)
* [Commit changes to your branch](#commit-changes-to-your-branch)
* [Submit a pull request](#submit-a-pull-request)
* [Code Review](#code-review)

# Introduction

You can contribute to this repos by opening a Pull Request. Direct push to the `master` branch is forbidden.

If your are new to git or github, the following articles may help you:

* See [Using Pull Requests](https://help.github.com/articles/using-pull-requests) for more information about Pull Requests.
* See [Fork A Repo](http://help.github.com/forking/) for an introduction to forking a repository.
* See [Creating branches](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) for an introduction on branching within GitHub.
* See [Refining patches using git](https://github.com/erlang/otp/wiki/Refining-patches-using-git) for an introduction to cleaning up git branches.

# Programming guidelines

When contributing to the library, for maintainability, ___please follow___ the 
[programming guidelines](https://www.mathworks.com/matlabcentral/fileexchange/46056-matlab-style-guidelines-2-0?s_tid=mwa_osa_a) 
for Matlab by Richard Johnson.

For information on object-oriented programming in Matlab, this quick 
[overview](http://www.cs.ubc.ca/~murphyk/Software/matlabTutorial/html/objectOriented.html)
by Kevin P Murphy is excellent.

A number of classes are defined as *handle* as opposed to *value* classes.
For more info on this important distinction, see the Matlab
[documentation.](https://www.mathworks.com/help/matlab/matlab_oop/comparing-handle-and-value-classes.html)

# Fixing a bug

* In most cases, pull requests for bug fixes should be based on the `master` branch.
* Indicate issue number in the commit (see commit section below)
* Do not close the issue yourself. The issue will be automatically closed when changes are pushed to master.

## Bug reporting template

For issues that are not straight forward, please use the template to structure the bug report:

    Title: [BUG] Summary of the issue. ex:"[BUG] sct_image command crashes when cropping image."

    Environment: Specify what OS and software version you are using.
    Step to reproduce: List all the steps that caused the issue.
    Expected results:
    Actual results:
    Data that caused the issue:

# Adding a new feature

* In most cases, pull requests for new features should be based on the master branch.
* It is important to write a good commit message explaining why the feature is needed. We prefer that the information is in the commit message, so that anyone that want to know two years later why a particular feature can easily find out. It does no harm to provide the same information in the pull request (if the pull request consists of a single commit, the commit message will be added to the pull request automatically).
* With few exceptions, it is mandatory to write a new test case that tests the feature. The test case is needed to ensure that the features does not stop working in the future.
* If you are implementing a new feature, also update the documentation to describe the feature.
* Make sure to cite any papers, algorithms or articles that can help understand the implementation of the feature.

## Feature request template

When proposing a new feature, a discussion will be conducted around the feature. Here a good way to present the new feature in the github issues.

    Title: [FEATURE] Summary of the feature.

    Motivation: Explain why the feature is needed.
    Use Case: Explain how the feature will be used, provide all the necessary steps.
    Expected Outcome: What will the outcome be.
    Citation: Provide references to any theoretical work to help the reader better understand the feature.

# Commit changes to your branch

Here are some tips to help the review go smoothly and quickly.

1. Keep it short. Keep the changes less then 50 lines.
2. Focus on committing 1 logical change at a time.
3. Write a verbose commit message. [Detailed explanation of a good commit message](https://github.com/erlang/otp/wiki/writing-good-commit-messages)
4. Correct any code style suggested by an analyser on your changes. [Matlab Code Checker](https://www.mathworks.com/help/fixedpoint/ug/using-the-matlab-code-analyzer-to-check-code-interactively-at-design-time.html).

## Commit message

### Title

The title should be short (50 chars or less), and should explicitly summarize the changes. If it solves an issue, add at the end: "fixes #ISSUE_NUMBER". The message should be preceded by one of the following flags:

```
BUG:   - a change made to fix a runtime issue (crash, segmentation fault, exception, incorrect result)
REF:   - refactoring (edits that don't impact the execution, renaming files, etc.)
OPT:   - a performance improvement, optimization, enhancement
BIN:   - any change related to binary files (should rarely be used)
NEW:   - new functionality added to the project (e.g., new function)
DOC:   - changes not related to the code (comments, documentation, etc.).
TEST:  - any change related to the testing (e.g., .travis, etc.)
```  

An example commit title might be:
```
BUG: Re-ordering of 4th dimension when apply transformation on 4D scans (fixes #1635)
````

### Description

```
Add more detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.

Further paragraphs come after blank lines.

  - Bullet points are okay, too

  - Typically a hyphen or asterisk is used for the bullet, preceded by a
    single space, with blank lines in between, but conventions vary here

Solves #1020
```

# Submit a pull request

### Title
The title should be short (50 chars or less), and should explicitly summarize the purpose of the PR.

### Labels
To help prioritize the request, add labels that describe the type and impact of the change. A change can have multiple types if it is appropriate but would have only 1 impact label. Such as `bug documentation fix:patch`.
See [label definitions](https://github.com/neuropoly/realtime_shimming/labels) on how to categorize the issue and pull request.

TODO: ADD AN EXAMPLE WITH LINK TO PR

# Code Review

[What is code review?](https://help.github.com/articles/about-pull-request-reviews/)

Any changes submitted to the master branch will go through code review. For a pull request to be accepted:

* At least 1 member should approve the changes.
* TravisCI must pass successfully

Reviewing members are:
* @jcohenadad
* @rtopfer
* @po09i 
