# Weazy: The Future of E-Commerce

Weazy (Web + Easy) is a next-generation, intelligent e-commerce platform powered by AI. It reimagines online shopping by replacing traditional search and browsing interfaces with a chat-based system that delivers a seamless, conversational experience — just like ChatGPT, but made for shopping.

Weazy is organized into three main repositories:

* **Frontend (`/front`)** — this interface (SPA built with React). [Weazy Frontend Repository](https://github.com/BassemArfaoui/weazy)
* **Backend (`/back`)** — REST API built with GoFiber for handling users, chats, wishlist, etc. [Weazy Backend Repository](https://github.com/BassemArfaoui/weazy-Server)
* **AI/ML API (`/api`)** — AI-powered services built with FastAPI for search, embeddings, and descriptions. [Weazy AI API Repository](https://github.com/BassemArfaoui/Weazy---API)

---

Users can search for products using **text**, **images**, or a **hybrid of both**. Under the hood, Weazy integrates multiple AI models to maximize accuracy and personalization, while giving sellers the tools to understand and target users more effectively.

---

## Key Features

* **Conversational Shopping Interface:** Ask for what you want using natural language or images.
* **Smart Search Options:**

  * **Text Search:** Semantic search using OpenAI's **CLIP**.
  * **Image Search:** Based on **ResNet50**, **VGG16**, or **CLIP** embeddings.
  * **Hybrid Search:** Combine text and image queries for more precision.
* **DeepSearch (Coming Soon):** Leverages multiple models and fusion strategies (e.g. averaging, max pooling) to return the most accurate results.
* **Recommendations (Coming Soon):** Personalized suggestions based on user activity like search history and wishlist items.
* **Wishlist System:** Save your favorite products for later.
* **Product Insights:** Generate AI-powered product descriptions on demand.
* **One-Click Redirection or Purchase (Coming Soon):** Either visit the original product page or complete purchases directly within Weazy.

---

## Architecture Overview

Weazy is divided into three main components:

* **Frontend (SPA with React)**: Provides a smooth user experience with chat-like interactions.
* **Backend API (GoFiber)**: Handles chats, users, wishlists, image uploads, and authentication.
* **AI/ML API (FastAPI)**: Powers all AI-related functionality such as search, recommendations, and description generation.

Additional services:

* **Cloudinary:** Hosts user-uploaded images.
* **PostgreSQL (NeonDB):** Stores structured data (users, chats, wishlist, products).
* **Milvus (Zilliz Cloud):** Hosts product embeddings and supports vector similarity search.

---

## Frontend (`/front`)

The frontend is built as a sleek, single-page application (SPA) using **React** and styled with **Tailwind CSS**, inspired by modern chat-based interfaces like ChatGPT and Grok.

### Tech Stack

* **React 19** with **React Router v7** for dynamic routing
* **TailwindCSS v4** for rapid, responsive UI styling
* **React Query v5** for smart data fetching and caching
* **Axios** for HTTP requests
* **Framer Motion** for smooth UI animations

### Key Features

* Seamless chat-like experience with real-time product search and interaction
* SPA design with logical navigation between chats, wishlist, and settings
* Frontend fetches data from both the **GoFiber API (backend)** and **FastAPI (ML/AI)**

### Development & Build

* Development: `npm run dev`
* Build: `npm run build`
* Preview: `npm run preview`

### Repository

* [Weazy Frontend Repository](https://github.com/BassemArfaoui/weazy)

---

## Backend (`/back`)

The backend is implemented using **GoFiber**, a fast and minimalist web framework for Go. It exposes RESTful endpoints that manage the core business logic and persistent data storage.

### Responsibilities

* User authentication and session management
* Chat creation, listing, renaming, and deletion
* Message saving and retrieval
* Wishlist item management
* Image uploads to Cloudinary

### Integrations

* **PostgreSQL** (via NeonDB) for all structured data
* **Cloudinary** for handling uploaded user images
* **Cross-service communication** with the FastAPI ML API for AI-related features

### Repository

* [Weazy Backend Repository](https://github.com/BassemArfaoui/weazy-Server)

---

## AI/ML API (`/api`)

The AI/ML layer is implemented using **FastAPI** and serves as the brain of the platform. It interprets user messages, determines the best tool for the request, and returns relevant responses.

### Responsibilities

* Process user queries (text, image, or hybrid)
* Select and apply the appropriate AI tool for search, description, or recommendation
* Interface with Milvus for vector similarity searches
* Manage embeddings and perform deep or hybrid search
* Generate product insights (e.g. descriptions)

### Integrations

* **CLIP**: Used for semantic search and hybrid matching
* **ResNet50** and **VGG16**: Image-based similarity matching
* **Milvus (Zilliz Cloud)**: Fast, scalable vector search database

### Repository

* [Weazy AI API Repository](https://github.com/BassemArfaoui/Weazy---API)
