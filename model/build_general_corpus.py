#!/usr/bin/env python3
"""
Expand conversations.txt with non-Minecraft training pairs.

The goal is to add a large amount of broad, useful, original training data
without filling the corpus with random noise. This generator focuses on:
- general definitions and explanations
- programming and web concepts
- science and math basics
- writing and communication help
- productivity and study habits
- geography capitals and countries
- arithmetic and percentage practice
"""

from __future__ import annotations

import argparse
import os
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA = os.path.join(BASE_DIR, 'data', 'general_expanded_corpus.txt')
MAIN_DATA = os.path.join(BASE_DIR, 'data', 'conversations.txt')


CONCEPTS = [
    ('python', 'Python is a general-purpose programming language known for readable syntax and broad library support.', 'It is useful for scripting, web services, automation, data work, and AI experiments.', 'A common example is writing a script to process files or call an API.'),
    ('javascript', 'JavaScript is a programming language used heavily for web interactivity and server-side development.', 'It matters because it runs in browsers and is widely used for modern applications.', 'A common example is handling button clicks or building an API with Node.js.'),
    ('typescript', 'TypeScript is JavaScript with optional static types and stronger tooling.', 'It helps teams catch mistakes earlier and improves editor support in larger codebases.', 'A common example is building a React or Node project with typed interfaces.'),
    ('html', 'HTML is the markup language used to structure web pages.', 'It matters because content needs structure before it can be styled or scripted.', 'A common example is defining headings, paragraphs, links, and forms.'),
    ('css', 'CSS is the language used to control the look and layout of web pages.', 'It matters because structure alone is not enough for readable or attractive interfaces.', 'A common example is changing fonts, colors, spacing, and responsive layouts.'),
    ('http', 'HTTP is the protocol used for communication between web clients and servers.', 'It matters because browsers, APIs, and many apps depend on request and response behavior.', 'A common example is a browser sending a GET request to load a page.'),
    ('https', 'HTTPS is HTTP protected by encryption through TLS.', 'It matters because it helps keep data private and prevents tampering in transit.', 'A common example is securely sending login credentials to a website.'),
    ('json', 'JSON is a text format for structured data using key-value pairs and arrays.', 'It matters because APIs and applications often exchange data in this format.', 'A common example is sending a response like {"ok": true, "name": "SolasAI"}.'),
    ('api', 'An API is a defined way for software systems to communicate with each other.', 'It matters because it lets one system use data or services from another safely and consistently.', 'A common example is an app calling a weather service for forecast data.'),
    ('database', 'A database is a system for storing and retrieving structured information.', 'It matters because apps need reliable persistence for users, content, and state.', 'A common example is saving accounts, messages, and settings in a relational database.'),
    ('sql', 'SQL is a language for querying and managing relational databases.', 'It matters because many production systems rely on relational storage and reporting.', 'A common example is selecting rows from a users table with a filter.'),
    ('git', 'Git is a version control system for tracking changes in files.', 'It matters because it supports collaboration, history, branching, and safer development workflows.', 'A common example is committing code before opening a pull request.'),
    ('github', 'GitHub is a platform for hosting repositories and collaborating on software projects.', 'It matters because it combines version control with code review, issues, and automation.', 'A common example is opening a pull request for a feature branch.'),
    ('debugging', 'Debugging is the process of finding and fixing problems in software.', 'It matters because software rarely works perfectly on the first try.', 'A common example is reproducing an error, narrowing the cause, and verifying a fix.'),
    ('algorithm', 'An algorithm is a step-by-step method for solving a problem.', 'It matters because good algorithms improve correctness and efficiency.', 'A common example is using binary search on sorted data.'),
    ('data structure', 'A data structure is a way of organizing data so operations are efficient and clear.', 'It matters because performance and code clarity depend on using the right structure.', 'A common example is using a hash map for quick lookups.'),
    ('array', 'An array is an ordered collection of values stored in sequence.', 'It matters because arrays are simple, common, and efficient for indexed access.', 'A common example is storing a list of numbers or messages in order.'),
    ('hash map', 'A hash map stores key-value pairs for fast lookup by key.', 'It matters because many problems need quick access to values by name or identifier.', 'A common example is mapping usernames to profile records.'),
    ('stack', 'A stack is a last-in, first-out collection.', 'It matters because call stacks, undo systems, and parsers often rely on this pattern.', 'A common example is pushing states as a user navigates through screens.'),
    ('queue', 'A queue is a first-in, first-out collection.', 'It matters because task scheduling and message handling often need ordered processing.', 'A common example is processing jobs in the order they arrive.'),
    ('function', 'A function is a reusable block of code that performs a task.', 'It matters because functions improve organization, reuse, and readability.', 'A common example is a helper that validates input before saving it.'),
    ('variable', 'A variable is a named place to store a value.', 'It matters because programs need ways to keep track of changing data.', 'A common example is storing a user name or score in a variable.'),
    ('loop', 'A loop repeats a block of code until a condition changes or a sequence ends.', 'It matters because repetitive tasks are common in software.', 'A common example is iterating over a list of records.'),
    ('conditional', 'A conditional chooses between actions based on a condition.', 'It matters because programs need decision-making logic.', 'A common example is showing an error message when a password is invalid.'),
    ('object oriented programming', 'Object-oriented programming organizes code around objects that combine data and behavior.', 'It matters because it can make larger systems easier to model and extend.', 'A common example is a User class with profile data and methods.'),
    ('functional programming', 'Functional programming emphasizes pure functions, immutability, and composition.', 'It matters because it can reduce side effects and improve predictability.', 'A common example is transforming a list with map and filter operations.'),
    ('class', 'A class is a blueprint for creating objects with fields and methods.', 'It matters because it groups related state and behavior together.', 'A common example is a Cart class with addItem and removeItem methods.'),
    ('object', 'An object is an instance containing data and behavior.', 'It matters because many languages model real systems as interacting objects.', 'A common example is an order object with status, items, and totals.'),
    ('exception', 'An exception is a signal that an error or unusual condition occurred.', 'It matters because robust programs need clear failure handling paths.', 'A common example is raising an exception when a file cannot be found.'),
    ('testing', 'Testing is the practice of verifying software behavior against expected outcomes.', 'It matters because automated checks reduce regressions and increase confidence.', 'A common example is asserting that a function returns the right result for known inputs.'),
    ('unit test', 'A unit test checks a small isolated part of a program.', 'It matters because it helps catch logic errors close to their source.', 'A common example is testing a parser with valid and invalid strings.'),
    ('integration test', 'An integration test verifies that multiple parts of a system work together.', 'It matters because many failures happen at boundaries between components.', 'A common example is testing an API route against a real database connection.'),
    ('refactoring', 'Refactoring is improving code structure without changing intended behavior.', 'It matters because clean code is easier to maintain and extend.', 'A common example is splitting a long function into smaller helpers.'),
    ('code review', 'Code review is the process of examining changes before they are merged.', 'It matters because another set of eyes often catches bugs and design issues.', 'A common example is reviewing a pull request for correctness and clarity.'),
    ('performance', 'Performance describes how fast and efficiently a system does work.', 'It matters because slow systems frustrate users and waste resources.', 'A common example is reducing an expensive query from seconds to milliseconds.'),
    ('memory leak', 'A memory leak happens when a program keeps memory it no longer needs.', 'It matters because leaks can degrade performance or crash long-running systems.', 'A common example is retaining objects in a cache that never evicts them.'),
    ('concurrency', 'Concurrency means making progress on multiple tasks during overlapping periods.', 'It matters because servers and modern apps often handle many requests or jobs at once.', 'A common example is processing background jobs while serving web traffic.'),
    ('thread', 'A thread is a unit of execution within a process.', 'It matters because threads can improve responsiveness or throughput in some workloads.', 'A common example is performing background work without blocking the main UI thread.'),
    ('process', 'A process is a running instance of a program with its own memory space.', 'It matters because operating systems isolate programs using processes.', 'A common example is a web server worker process handling requests.'),
    ('operating system', 'An operating system manages hardware, files, processes, and basic system services.', 'It matters because software runs on top of it and relies on its abstractions.', 'A common example is Linux managing files, memory, and network sockets.'),
    ('linux', 'Linux is an open-source operating system family used on servers, desktops, and embedded systems.', 'It matters because it is widely used for development and infrastructure.', 'A common example is deploying a web app to a Linux server.'),
    ('windows', 'Windows is a widely used operating system developed by Microsoft.', 'It matters because many desktop applications and enterprise environments rely on it.', 'A common example is running office software or development tools on a Windows laptop.'),
    ('macos', 'macOS is Apple’s desktop operating system.', 'It matters because many developers and creative professionals use it daily.', 'A common example is building software on a MacBook with Unix-style tools.'),
    ('file system', 'A file system organizes how data is stored and retrieved on disk.', 'It matters because applications depend on predictable file and directory behavior.', 'A common example is reading logs from a folder structure.'),
    ('command line', 'The command line is a text interface for interacting with the computer.', 'It matters because it enables scripting, automation, and fast workflows.', 'A common example is running a build script or searching files with a shell command.'),
    ('shell script', 'A shell script is a file containing commands executed by a shell.', 'It matters because it automates repetitive system tasks.', 'A common example is a deployment script that installs dependencies and restarts services.'),
    ('regular expression', 'A regular expression is a pattern used to match or transform text.', 'It matters because text validation and extraction are common tasks.', 'A common example is finding email-like strings in a log file.'),
    ('encryption', 'Encryption transforms data so only authorized parties can read it.', 'It matters because privacy and secure communication depend on it.', 'A common example is HTTPS encrypting traffic between a browser and server.'),
    ('authentication', 'Authentication verifies who a user or system claims to be.', 'It matters because access control starts with identity.', 'A common example is logging in with a password or token.'),
    ('authorization', 'Authorization decides what an authenticated user is allowed to do.', 'It matters because not every logged-in user should have the same permissions.', 'A common example is allowing admins to manage users while regular users cannot.'),
    ('cookie', 'A cookie is small data stored by a browser and associated with a site.', 'It matters because sessions, preferences, and tracking often rely on cookies.', 'A common example is keeping a user signed in after login.'),
    ('session', 'A session represents ongoing interaction state between a client and server.', 'It matters because many applications need continuity across requests.', 'A common example is remembering a logged-in user across page loads.'),
    ('cache', 'A cache stores data temporarily so future access is faster.', 'It matters because recomputing or refetching everything can be expensive.', 'A common example is caching frequent database query results.'),
    ('latency', 'Latency is the delay before a response starts after a request is made.', 'It matters because users notice responsiveness quickly.', 'A common example is the time between clicking a button and seeing data load.'),
    ('bandwidth', 'Bandwidth describes how much data can be transferred over time.', 'It matters because data-heavy systems depend on transfer capacity.', 'A common example is video streaming quality depending partly on bandwidth.'),
    ('dns', 'DNS translates domain names into IP addresses.', 'It matters because people use names while networks route using addresses.', 'A common example is converting example.com into the server IP behind it.'),
    ('ip address', 'An IP address identifies a device or service on a network.', 'It matters because network communication needs addressing.', 'A common example is a server listening on a public IP address.'),
    ('router', 'A router directs network traffic between devices and networks.', 'It matters because homes, offices, and data centers need traffic routing.', 'A common example is a home router connecting Wi‑Fi devices to the internet.'),
    ('cloud computing', 'Cloud computing is using remote infrastructure and services over the internet.', 'It matters because it lets teams deploy systems without owning all the hardware.', 'A common example is running a web app on a managed cloud platform.'),
    ('virtual machine', 'A virtual machine is a software-based computer running inside another system.', 'It matters because it provides isolation and flexible infrastructure.', 'A common example is hosting separate application environments on one server.'),
    ('container', 'A container packages software with its runtime dependencies in an isolated unit.', 'It matters because it improves reproducibility across environments.', 'A common example is shipping a web app in a Docker container.'),
    ('docker', 'Docker is a platform for building and running containers.', 'It matters because it simplifies packaging software consistently.', 'A common example is starting a database locally with a container image.'),
    ('ci cd', 'CI/CD refers to automated integration, testing, and delivery workflows.', 'It matters because it reduces manual release work and catches issues earlier.', 'A common example is running tests automatically on every push and deploying after approval.'),
    ('machine learning', 'Machine learning is a way for computers to learn patterns from data.', 'It matters because many tasks are easier to solve with learned patterns than explicit rules.', 'A common example is training a model to classify text or images.'),
    ('deep learning', 'Deep learning is a branch of machine learning based on layered neural networks.', 'It matters because it is powerful for tasks like vision, speech, and language.', 'A common example is using a transformer model for text generation.'),
    ('neural network', 'A neural network is a model made of layered mathematical transformations that learn from examples.', 'It matters because it can approximate complex patterns in data.', 'A common example is predicting the next token in a sentence.'),
    ('transformer', 'A transformer is a neural network architecture that uses attention to model sequences.', 'It matters because modern language models rely on this design.', 'A common example is generating text conditioned on a prompt.'),
    ('token', 'A token is a piece of text used as a unit by a language model.', 'It matters because models read and generate text as token sequences.', 'A common example is splitting a prompt into subword tokens before inference.'),
    ('embedding', 'An embedding is a numeric representation that captures useful relationships in data.', 'It matters because systems use embeddings for search, similarity, and retrieval.', 'A common example is comparing two sentences by embedding distance.'),
    ('vector', 'A vector is an ordered list of numbers.', 'It matters because vectors represent features, positions, directions, and learned meanings.', 'A common example is storing an embedding as a high-dimensional vector.'),
    ('training', 'Training is the process of updating model parameters using data and an objective.', 'It matters because model quality depends heavily on the training procedure and dataset.', 'A common example is minimizing loss over many batches of text data.'),
    ('inference', 'Inference is using a trained model to produce outputs from new inputs.', 'It matters because deployment depends on inference speed and quality.', 'A common example is generating an answer to a user prompt.'),
    ('overfitting', 'Overfitting happens when a model learns the training data too narrowly and generalizes poorly.', 'It matters because strong training performance alone does not guarantee useful real-world behavior.', 'A common example is a model that memorizes examples but fails on new ones.'),
    ('underfitting', 'Underfitting happens when a model is too weak or too poorly trained to learn the main pattern.', 'It matters because the model then performs badly on both training and new data.', 'A common example is using an overly simple model for a complex classification task.'),
    ('loss', 'Loss is a number measuring how wrong a model’s predictions are.', 'It matters because optimization tries to reduce loss during training.', 'A common example is cross-entropy loss for language modeling.'),
    ('gradient descent', 'Gradient descent is an optimization method that updates parameters to reduce loss.', 'It matters because many machine learning systems learn through repeated gradient-based updates.', 'A common example is backpropagation combined with AdamW optimizer steps.'),
    ('attention', 'Attention is a mechanism that lets a model focus on the most relevant parts of its input.', 'It matters because it improves how sequence models capture long-range relationships.', 'A common example is a token attending to earlier tokens that define its meaning.'),
    ('prompt', 'A prompt is the input instruction or context given to a model.', 'It matters because model output depends heavily on how the request is framed.', 'A common example is asking for a short summary with a specific tone.'),
    ('retrieval', 'Retrieval means fetching relevant information before producing an answer.', 'It matters because a model can respond more accurately when grounded in relevant data.', 'A common example is searching documentation before answering a user question.'),
    ('rag', 'RAG stands for retrieval-augmented generation.', 'It matters because it combines search and generation to produce more grounded responses.', 'A common example is answering a question using relevant passages from an internal knowledge base.'),
    ('probability', 'Probability measures how likely an event is.', 'It matters because uncertainty appears in science, statistics, and machine learning.', 'A common example is estimating the chance of rain tomorrow.'),
    ('statistics', 'Statistics is the study of collecting, summarizing, and interpreting data.', 'It matters because data-driven decisions depend on careful analysis.', 'A common example is estimating an average and understanding the spread around it.'),
    ('mean', 'The mean is the arithmetic average of a set of values.', 'It matters because it summarizes a central tendency of data.', 'A common example is averaging test scores across a class.'),
    ('median', 'The median is the middle value when data is ordered.', 'It matters because it is often more robust than the mean when outliers exist.', 'A common example is reporting the median salary in a skewed dataset.'),
    ('standard deviation', 'Standard deviation measures how spread out data is around the mean.', 'It matters because it describes variability, not just central value.', 'A common example is comparing consistency between two sets of measurements.'),
    ('physics', 'Physics is the study of matter, energy, motion, and forces.', 'It matters because it helps explain how the physical world behaves.', 'A common example is predicting how gravity affects a falling object.'),
    ('chemistry', 'Chemistry studies matter, atoms, molecules, and how substances change.', 'It matters because materials, reactions, and life processes all depend on chemistry.', 'A common example is understanding why rust forms on iron.'),
    ('biology', 'Biology is the study of living organisms and life processes.', 'It matters because health, ecosystems, and evolution all depend on biological systems.', 'A common example is studying how cells convert nutrients into energy.'),
    ('cell', 'A cell is the basic structural and functional unit of life.', 'It matters because every organism is built from cells or cell-like structures.', 'A common example is a human blood cell carrying oxygen.'),
    ('dna', 'DNA is the molecule that stores genetic information in living organisms.', 'It matters because heredity and biological development depend on it.', 'A common example is traits being passed from parents to children through genes.'),
    ('evolution', 'Evolution is the change in inherited traits across populations over generations.', 'It matters because it explains biodiversity and adaptation.', 'A common example is a population changing over time under environmental pressures.'),
    ('gravity', 'Gravity is the force by which masses attract each other.', 'It matters because it shapes motion on Earth and across the universe.', 'A common example is objects falling toward the ground.'),
    ('energy', 'Energy is the capacity to do work or cause change.', 'It matters because every physical process involves energy transfer or transformation.', 'A common example is electrical energy powering a light bulb.'),
    ('ecosystem', 'An ecosystem is a community of organisms interacting with each other and their environment.', 'It matters because living systems depend on balanced relationships and resources.', 'A common example is a forest with plants, animals, fungi, water, and soil.'),
    ('economics', 'Economics studies choices about resources, production, and exchange.', 'It matters because scarcity and incentives shape real-world behavior.', 'A common example is analyzing how prices change with supply and demand.'),
    ('inflation', 'Inflation is a general increase in prices over time.', 'It matters because it affects purchasing power and financial planning.', 'A common example is the same amount of money buying fewer groceries than before.'),
    ('budget', 'A budget is a plan for income, spending, and saving.', 'It matters because financial goals are easier to reach with a clear plan.', 'A common example is setting monthly limits for rent, food, and savings.'),
    ('interest', 'Interest is the cost of borrowing money or the reward for lending it.', 'It matters because loans and savings both depend on interest over time.', 'A common example is a bank paying interest on a savings account.'),
    ('productivity', 'Productivity is the ability to complete meaningful work efficiently.', 'It matters because time and attention are limited.', 'A common example is batching similar tasks and reducing distractions.'),
    ('time management', 'Time management is planning and using time intentionally.', 'It matters because priorities compete for limited hours.', 'A common example is scheduling focused work blocks and breaks.'),
    ('goal setting', 'Goal setting is defining desired outcomes clearly enough to work toward them.', 'It matters because vague intentions are harder to act on consistently.', 'A common example is turning “study more” into “study math for 45 minutes daily.”'),
    ('note taking', 'Note taking is recording key ideas so they can be reviewed and used later.', 'It matters because memory is limited and understanding improves through active capture.', 'A common example is summarizing a lecture in your own words.'),
    ('writing', 'Writing is the process of communicating ideas through structured language.', 'It matters because clear writing improves understanding and persuasion.', 'A common example is drafting an email that gets to the point quickly.'),
    ('editing', 'Editing is revising writing to improve clarity, structure, tone, and correctness.', 'It matters because first drafts often contain confusion or unnecessary detail.', 'A common example is cutting repetition and sharpening the main point of a paragraph.'),
    ('summary', 'A summary is a short version of a larger text focusing on the main points.', 'It matters because it helps readers understand the core idea quickly.', 'A common example is condensing a long article into a few sentences.'),
    ('argument', 'An argument is a claim supported by reasons and evidence.', 'It matters because reasoning needs support to be convincing.', 'A common example is arguing for a design decision by discussing tradeoffs and evidence.'),
    ('evidence', 'Evidence is information used to support or challenge a claim.', 'It matters because good decisions and arguments depend on support, not just opinion.', 'A common example is using data from tests to justify a bug fix.'),
    ('critical thinking', 'Critical thinking is evaluating claims and reasoning carefully before accepting them.', 'It matters because not every confident statement is accurate or well-supported.', 'A common example is checking assumptions before trusting a conclusion.'),
    ('communication', 'Communication is the exchange of information and meaning between people or systems.', 'It matters because coordination depends on shared understanding.', 'A common example is explaining a technical issue so a teammate can act on it.'),
    ('listening', 'Listening is actively paying attention to understand what someone is really saying.', 'It matters because miscommunication often comes from rushing to reply instead of understanding.', 'A common example is clarifying a requirement before implementing it.'),
    ('leadership', 'Leadership is helping a group move effectively toward a goal.', 'It matters because teams need direction, prioritization, and trust.', 'A common example is setting clear priorities during a difficult release.'),
    ('teamwork', 'Teamwork is coordinating effort with others toward a shared outcome.', 'It matters because many goals are too large or complex for one person alone.', 'A common example is dividing tasks while maintaining shared context.'),
    ('learning', 'Learning is the process of gaining knowledge, skills, or understanding.', 'It matters because improvement depends on practice, feedback, and reflection.', 'A common example is studying a concept, applying it, and reviewing mistakes.'),
    ('habit', 'A habit is a behavior repeated often enough to become automatic or easier to trigger.', 'It matters because consistent systems often beat occasional motivation.', 'A common example is reviewing notes at the same time every day.'),
    ('focus', 'Focus is sustained attention on a chosen task.', 'It matters because context switching can reduce speed and quality.', 'A common example is silencing notifications during a deep work block.'),
    ('stress', 'Stress is the body and mind’s response to pressure or challenge.', 'It matters because too much unmanaged stress hurts performance and wellbeing.', 'A common example is feeling tense when deadlines pile up without a plan.'),
    ('sleep', 'Sleep is a biological process essential for recovery, memory, and health.', 'It matters because poor sleep affects attention, mood, and learning.', 'A common example is studying less effectively after a short night of sleep.'),
    ('exercise', 'Exercise is planned physical activity that improves health and fitness.', 'It matters because movement supports energy, strength, and long-term wellbeing.', 'A common example is taking a brisk walk or doing resistance training regularly.'),
    ('nutrition', 'Nutrition is how the body uses food for energy, growth, and maintenance.', 'It matters because health and performance depend on consistent intake of needed nutrients.', 'A common example is balancing meals with protein, fiber, and hydration.'),
    ('internet', 'The internet is a global network of connected computers and services.', 'It matters because modern communication, software, and information access depend on it.', 'A common example is sending messages or loading a website over the internet.'),
    ('browser', 'A browser is software used to access and display web content.', 'It matters because many applications are now delivered through the web.', 'A common example is using a browser to visit documentation or email.'),
    ('search engine', 'A search engine indexes content and helps users find relevant information.', 'It matters because large information spaces need ranking and retrieval.', 'A common example is searching for a tutorial or troubleshooting guide.'),
    ('open source', 'Open source software makes its source code available for inspection, modification, and reuse under a license.', 'It matters because it enables collaboration, transparency, and shared tools.', 'A common example is contributing a fix to a public library.'),
    ('license', 'A license defines how software or content may be used, shared, and modified.', 'It matters because permissions and obligations vary by license type.', 'A common example is choosing an MIT or Apache license for a project.'),
]

CAPITALS = [
    ('france', 'paris'), ('germany', 'berlin'), ('italy', 'rome'), ('spain', 'madrid'), ('portugal', 'lisbon'),
    ('netherlands', 'amsterdam'), ('belgium', 'brussels'), ('switzerland', 'bern'), ('austria', 'vienna'), ('poland', 'warsaw'),
    ('czech republic', 'prague'), ('hungary', 'budapest'), ('greece', 'athens'), ('turkey', 'ankara'), ('sweden', 'stockholm'),
    ('norway', 'oslo'), ('denmark', 'copenhagen'), ('finland', 'helsinki'), ('ireland', 'dublin'), ('united kingdom', 'london'),
    ('canada', 'ottawa'), ('united states', 'washington, d.c.'), ('mexico', 'mexico city'), ('brazil', 'brasília'), ('argentina', 'buenos aires'),
    ('chile', 'santiago'), ('colombia', 'bogotá'), ('peru', 'lima'), ('venezuela', 'caracas'), ('australia', 'canberra'),
    ('new zealand', 'wellington'), ('japan', 'tokyo'), ('south korea', 'seoul'), ('china', 'beijing'), ('india', 'new delhi'),
    ('pakistan', 'islamabad'), ('indonesia', 'jakarta'), ('thailand', 'bangkok'), ('vietnam', 'hanoi'), ('philippines', 'manila'),
    ('singapore', 'singapore'), ('malaysia', 'kuala lumpur'), ('saudi arabia', 'riyadh'), ('united arab emirates', 'abu dhabi'), ('egypt', 'cairo'),
    ('south africa', 'pretoria'), ('kenya', 'nairobi'), ('nigeria', 'abuja'), ('morocco', 'rabat'), ('ethiopia', 'addis ababa'),
]

UNITS = [
    ('1 kilometer', '1000 meters'), ('1 meter', '100 centimeters'), ('1 centimeter', '10 millimeters'),
    ('1 kilogram', '1000 grams'), ('1 liter', '1000 milliliters'), ('1 hour', '60 minutes'),
    ('1 minute', '60 seconds'), ('1 day', '24 hours'), ('1 week', '7 days'), ('1 year', '12 months'),
    ('1 byte', '8 bits'), ('1 inch', '2.54 centimeters'), ('1 foot', '12 inches'), ('1 yard', '3 feet'),
    ('1 mile', '5280 feet'), ('1 pound', '16 ounces'), ('1 gallon', '128 fluid ounces'),
]

SCHOOL_SUBJECTS = [
    ('algebra', 'Algebra uses symbols and equations to describe relationships between quantities.', 'It matters because it helps solve unknowns and model patterns.', 'A common example is solving x + 7 = 12.'),
    ('geometry', 'Geometry studies shapes, sizes, positions, and spatial relationships.', 'It matters because measurement and design depend on spatial reasoning.', 'A common example is finding the area of a rectangle.'),
    ('trigonometry', 'Trigonometry studies relationships between angles and sides of triangles.', 'It matters because it is used in physics, engineering, and navigation.', 'A common example is using sine to find a missing side.'),
    ('calculus', 'Calculus studies change, motion, and accumulation.', 'It matters because many advanced models in science and engineering depend on it.', 'A common example is using derivatives to measure rate of change.'),
    ('history', 'History is the study of past events and how they shape the present.', 'It matters because context helps explain why institutions, conflicts, and cultures developed the way they did.', 'A common example is studying the causes and effects of a revolution.'),
    ('geography', 'Geography studies places, environments, and how humans interact with them.', 'It matters because climate, resources, and location strongly influence societies.', 'A common example is comparing how rivers affect settlement patterns.'),
    ('literature', 'Literature studies written works such as novels, poems, and plays.', 'It matters because reading closely improves interpretation, empathy, and communication.', 'A common example is analyzing theme and character development in a novel.'),
    ('grammar', 'Grammar is the system of rules and patterns used to organize language.', 'It matters because clear sentences depend on understandable structure.', 'A common example is choosing the right verb tense in a paragraph.'),
    ('astronomy', 'Astronomy is the study of stars, planets, galaxies, and the universe.', 'It matters because it helps explain cosmic structure, motion, and origins.', 'A common example is studying how planets orbit stars.'),
    ('geology', 'Geology studies rocks, Earth structure, and processes that shape the planet.', 'It matters because landscapes, resources, and hazards are tied to geological change.', 'A common example is explaining how volcanoes form.'),
]

CODING_HELP = [
    ('how do i debug a python error', 'Read the traceback carefully, isolate the failing line, check the values going into it, and reproduce the problem with the smallest possible example.'),
    ('how do i fix a syntax error', 'Start at the exact line shown by the parser, then check the line before it for missing brackets, quotes, commas, or punctuation.'),
    ('how do i learn programming', 'Start with one language, build small projects, practice consistently, and debug your own mistakes instead of only reading tutorials.'),
    ('how do i get better at coding', 'Write code often, finish small programs, review why bugs happened, and learn common patterns instead of memorizing isolated tricks.'),
    ('how do i read code better', 'Trace the main data flow first, identify functions and inputs, and then follow one path through the program at a time.'),
    ('how do i write cleaner code', 'Use clear names, keep functions focused, remove repetition, and organize logic so each part has one obvious job.'),
    ('how do i choose a data structure', 'Start from the operations you need most often, then choose the structure that makes those operations simple and efficient.'),
    ('how do i prepare for a coding interview', 'Practice core data structures, explain your thinking out loud, solve timed problems, and review mistakes instead of just collecting solutions.'),
    ('how do i write an api', 'Define the inputs and outputs clearly, keep routes consistent, validate requests, and return errors that are easy for clients to understand.'),
    ('how do i make a website responsive', 'Use flexible layouts, relative sizing, and breakpoints so the design adapts cleanly to smaller screens.'),
    ('how do i fix a slow query', 'Measure first, inspect the execution pattern, reduce unnecessary work, and add indexes or better filters only after understanding the bottleneck.'),
    ('how do i write better tests', 'Test the behavior that matters, use clear inputs and expected outputs, and focus on bugs and regressions the code is likely to have.'),
    ('how do i review code well', 'Look for correctness, edge cases, maintainability, and missing tests before focusing on style details.'),
    ('how do i design a project', 'Start with the main user flow, define data boundaries, keep modules small, and avoid complexity you cannot justify yet.'),
    ('how do i learn web development', 'Start with HTML, CSS, and JavaScript, then build simple pages, forms, and APIs before adding frameworks.'),
]

CONVERSATIONAL_HELP = [
    ('im bored', 'Try picking one concrete thing to learn, build, or improve instead of waiting for motivation to appear on its own.'),
    ('what should i learn today', 'Pick one small useful topic, spend focused time on it, and finish with a short review of what you understood and what still feels weak.'),
    ('i feel stuck', 'Reduce the problem until the next step feels obvious, then solve only that step instead of wrestling with the whole thing at once.'),
    ('i do not understand this', 'Tell me which exact part is confusing and I will break it down into smaller pieces.'),
    ('can you explain that another way', 'Yes. I can restate it more simply, compare it to an example, or turn it into step-by-step instructions.'),
    ('i keep making mistakes', 'That usually means you are close to learning something important. Look for the pattern in the mistakes instead of treating each one as random.'),
    ('how do i stay motivated', 'Use small goals, visible progress, and regular practice. Motivation is unreliable, but habits and clear next steps are steadier.'),
    ('how do i ask better questions', 'State the goal, show what you tried, include the relevant details, and explain exactly where the confusion begins.'),
    ('how do i stop procrastinating', 'Shrink the task, start a short timer, remove distractions, and commit only to the first useful step instead of the whole job.'),
    ('how do i learn from mistakes', 'Review what happened, identify the wrong assumption, and change the process so the same failure is less likely next time.'),
]


CONCEPT_QUESTION_TEMPLATES = [
    'what is {name}',
    'explain {name}',
    'explain {name} simply',
    'what does {name} mean',
    'why does {name} matter',
    'when would i use {name}',
    'give me an example of {name}',
    'teach me about {name}',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=OUTPUT_DATA)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge-target', type=str, default=MAIN_DATA)
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def parse_existing_pairs(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    pairs: set[tuple[str, str]] = set()
    pending_user = None
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith('User: '):
                pending_user = line[6:].strip()
            elif line.startswith('Assistant: ') and pending_user:
                pairs.add((pending_user, line[11:].strip()))
                pending_user = None
    return pairs


def write_pairs(path: str, pairs: list[tuple[str, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for question, answer in pairs:
            handle.write(f'User: {question}\n')
            handle.write(f'Assistant: {answer}\n')


def append_unique_pairs(path: str, pairs: list[tuple[str, str]]) -> int:
    existing = parse_existing_pairs(path)
    to_add = [(q, a) for q, a in pairs if (q, a) not in existing]
    if not to_add:
        return 0
    with open(path, 'a', encoding='utf-8') as handle:
        for question, answer in to_add:
            handle.write(f'\nUser: {question}\n')
            handle.write(f'Assistant: {answer}\n')
    return len(to_add)


def concept_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, definition, importance, example in CONCEPTS:
        answers = {
            'what is': definition,
            'explain': f'{definition} {importance}',
            'explain simply': f'{definition}',
            'what does': definition,
            'why does': importance,
            'when would i use': f'{importance} {example}',
            'give me an example of': example,
            'teach me about': f'{definition} {importance} {example}',
        }
        for template in CONCEPT_QUESTION_TEMPLATES:
            question = clean_text(template.format(name=name))
            for key, answer in answers.items():
                if template.startswith(key):
                    pairs.append((question, clean_text(answer)))
                    break
        pairs.append((clean_text(f'compare {name} to a beginner explanation'), clean_text(f'{definition} {importance} {example}')))
        pairs.append((clean_text(f'why should i learn {name}'), clean_text(f'{importance} {example}')))
        pairs.append((clean_text(f'how do i understand {name} better'), clean_text(f'Start with the core idea: {definition} Then look at why it matters: {importance} Finally, connect it to practice: {example}')))
    return pairs


def capital_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for country, capital in CAPITALS:
        pairs.append((f'what is the capital of {country}', f'The capital of {country.title()} is {capital.title()}.' if capital != 'washington, d.c.' else 'The capital of the United States is Washington, D.C.'))
        pairs.append((f'capital of {country}', f'{capital.title()} is the capital of {country.title()}.' if capital != 'washington, d.c.' else 'Washington, D.C. is the capital of the United States.'))
    return [(clean_text(q), clean_text(a)) for q, a in pairs]


def unit_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for left, right in UNITS:
        pairs.append((f'convert {left}', f'{left.title()} equals {right}.'))
        pairs.append((f'how many in {left}', f'{left.title()} is {right}.'))
    return [(clean_text(q), clean_text(a)) for q, a in pairs]


def arithmetic_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a in range(1, 51):
        for b in range(1, 21):
            pairs.append((f'what is {a} plus {b}', f'{a} plus {b} equals {a + b}.'))
            pairs.append((f'what is {a} minus {b}', f'{a} minus {b} equals {a - b}.'))
            pairs.append((f'what is {a} times {b}', f'{a} times {b} equals {a * b}.'))
            if a * b <= 400:
                pairs.append((f'what is {a * b} divided by {b}', f'{a * b} divided by {b} equals {a}.'))
    for base in range(1, 101):
        pairs.append((f'what is the square of {base}', f'The square of {base} is {base * base}.'))
        pairs.append((f'what is 10 percent of {base}', f'10 percent of {base} is {base / 10:g}.'))
        pairs.append((f'what is 25 percent of {base}', f'25 percent of {base} is {base / 4:g}.'))
        pairs.append((f'what is 50 percent of {base}', f'50 percent of {base} is {base / 2:g}.'))
    return [(clean_text(q), clean_text(a)) for q, a in pairs]


def writing_pairs() -> list[tuple[str, str]]:
    prompts = [
        ('how do i write a good paragraph', 'Start with one clear main idea, support it with useful details, and end cleanly without wandering.'),
        ('how do i write a better email', 'State the purpose early, keep the wording direct, include the needed details, and end with a clear next step.'),
        ('how do i summarize an article', 'Find the main claim, keep only the key supporting points, and rewrite them briefly in your own words.'),
        ('how do i study better', 'Use short focused sessions, practice active recall, review mistakes, and come back to the material over time.'),
        ('how do i focus better', 'Reduce distractions, set one concrete task, work in a timed block, and take short breaks before attention collapses.'),
        ('how do i manage my time', 'Choose priorities first, estimate effort honestly, schedule focused work blocks, and review what slipped.'),
        ('how do i learn faster', 'Break the topic into parts, practice actively, seek feedback, and revisit weak areas instead of rereading passively.'),
        ('how do i explain something clearly', 'Start with the main idea, define important terms simply, and then connect them to one practical example.'),
        ('how do i make an argument stronger', 'State the claim clearly, support it with evidence, address objections, and avoid exaggeration.'),
        ('how do i edit my writing', 'Cut repetition, tighten weak sentences, improve structure, and check whether each paragraph serves the main point.'),
    ]
    pairs: list[tuple[str, str]] = []
    for q, a in prompts:
        pairs.append((q, a))
        pairs.append((q.replace('how do i', 'best way to'), a))
        pairs.append((q.replace('how do i', 'help me'), a))
        pairs.append((q.replace('how do i', 'teach me how to'), a))
    return [(clean_text(q), clean_text(a)) for q, a in pairs]


def school_subject_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, definition, importance, example in SCHOOL_SUBJECTS:
        pairs.append((f'what is {name}', definition))
        pairs.append((f'explain {name}', f'{definition} {importance}'))
        pairs.append((f'why does {name} matter', importance))
        pairs.append((f'give me an example of {name}', example))
        pairs.append((f'help me understand {name}', f'Start with the core idea: {definition} Then connect it to why it matters: {importance} {example}'))
    return [(clean_text(q), clean_text(a)) for q, a in pairs]


def coding_help_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for question, answer in CODING_HELP:
        pairs.append((question, answer))
        pairs.append((question.replace('how do i', 'best way to'), answer))
        pairs.append((question.replace('how do i', 'help me'), answer))
    return [(clean_text(q), clean_text(a)) for q, a in pairs]


def supportive_conversation_pairs() -> list[tuple[str, str]]:
    return [(clean_text(q), clean_text(a)) for q, a in CONVERSATIONAL_HELP]


def conversation_pairs() -> list[tuple[str, str]]:
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'yo', 'yoo', 'sup']
    greeting_answers = [
        'Hello. What would you like help with?',
        'Hi. Ask me a question and I will answer clearly.',
        'Hey. Tell me what you want to learn or figure out.',
    ]
    confirmations = ['okay', 'ok', 'sounds good', 'nice', 'cool', 'great']
    confirm_answers = [
        'All right. What should we work on next?',
        'Good. Ask another question whenever you want.',
        'Understood. Give me the next topic.'
    ]
    pairs: list[tuple[str, str]] = []
    for q in greetings:
        for a in greeting_answers:
            pairs.append((q, a))
    for q in confirmations:
        for a in confirm_answers:
            pairs.append((q, a))
    return pairs


def generate_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pairs.extend(concept_pairs())
    pairs.extend(capital_pairs())
    pairs.extend(unit_pairs())
    pairs.extend(arithmetic_pairs())
    pairs.extend(writing_pairs())
    pairs.extend(school_subject_pairs())
    pairs.extend(coding_help_pairs())
    pairs.extend(conversation_pairs())
    pairs.extend(supportive_conversation_pairs())
    return list(dict.fromkeys((clean_text(q), clean_text(a)) for q, a in pairs if clean_text(q) and clean_text(a)))


def main() -> int:
    args = parse_args()
    pairs = generate_pairs()
    write_pairs(args.output, pairs)
    print(f'General corpus pairs written: {len(pairs)} -> {args.output}')
    if args.merge:
        added = append_unique_pairs(args.merge_target, pairs)
        print(f'Merged unique pairs: {added} -> {args.merge_target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())