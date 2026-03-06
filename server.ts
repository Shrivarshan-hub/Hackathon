import express from "express";
import { createServer as createViteServer } from "vite";
import Database from "better-sqlite3";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import rateLimit from "express-rate-limit";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { randomUUID } from "crypto";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const JWT_SECRET = process.env.JWT_SECRET || "shethrive-dev-secret-change-in-production";
const SALT_ROUNDS = 12;
const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
};

const db = new Database("shethrive.db");

// Schema: users (UUID, full profile), chats/messages (user_id as TEXT uuid)
// Drop and recreate users so new auth schema applies; chats/messages recreated for user_id type
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    age INTEGER NOT NULL,
    career_field TEXT NOT NULL,
    experience_years INTEGER NOT NULL,
    current_role TEXT,
    salary_range TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    role TEXT,
    content TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  );
`);

// Password validation: min 8 chars, 1 upper, 1 number, 1 special
function validatePassword(p: string): { ok: boolean; message?: string } {
  if (p.length < 8) return { ok: false, message: "Password must be at least 8 characters" };
  if (!/[A-Z]/.test(p)) return { ok: false, message: "Password must contain at least one uppercase letter" };
  if (!/[0-9]/.test(p)) return { ok: false, message: "Password must contain at least one number" };
  if (!/[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(p)) return { ok: false, message: "Password must contain at least one special character" };
  return { ok: true };
}

// Sanitize string inputs
function sanitize(s: unknown): string {
  if (typeof s !== "string") return "";
  return s.trim().slice(0, 500);
}

interface JwtPayload {
  userId: string;
  email: string;
}

function authMiddleware(req: express.Request, res: express.Response, next: express.NextFunction) {
  const token = req.cookies?.token;
  if (!token) {
    return res.status(401).json({ error: "Authentication required" });
  }
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as JwtPayload;
    const user = db.prepare(
      "SELECT id, full_name, email, age, career_field, experience_years, current_role, salary_range, created_at FROM users WHERE id = ?"
    ).get(decoded.userId) as any;
    if (!user) return res.status(401).json({ error: "User not found" });
    req.user = { ...user, name: user.full_name };
    next();
  } catch {
    res.clearCookie("token", COOKIE_OPTIONS);
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}

declare global {
  namespace Express {
    interface Request {
      user?: { id: string; name: string; email: string;[k: string]: any };
    }
  }
}

const SYSTEM_INSTRUCTIONS: Record<string, string> = {
  coach:
    "You are SheThrive, an AI career coach for women. Your goal is to help women negotiate smarter, grow faster, and own their worth. Be encouraging, professional, and provide actionable advice. Focus on closing the gender pay gap and overcoming systemic barriers.",
  negotiate:
    "You are a Hiring Manager in a salary negotiation simulation. Be realistic, slightly firm but professional. Start by offering a base salary that is slightly below market rate. Respond to the user's negotiation attempts based on their arguments.",
  salary:
    "You are a salary research expert. Provide detailed market benchmarks, salary ranges, and tips for researching specific roles and industries. Use tables where appropriate.",
  mentor:
    "You are a career mentor. Focus on long-term growth, networking strategies, and leadership development for women in tech and corporate environments.",
  progress:
    "You are a career progress tracker. Help the user set goals, break them down into weekly tasks, and provide accountability.",
  recruiter:
    "You are a tech recruiter. Conduct a realistic mock interview or screening call. Focus on how the user presents their value and achievements.",
};

const API_KEY = process.env.API_KEY || process.env.GEMINI_API_KEY || "";

async function sendMessage(message: string, history: any[], mode: string): Promise<string> {
  if (!API_KEY) {
    throw new Error("API key is not set. Please add API_KEY (or GEMINI_API_KEY) to your .env file.");
  }

  const contents = [
    ...history.map((m: any) => ({
      role: m.role === "user" ? "user" : "model",
      parts: [{ text: m.content }],
    })),
    { role: "user", parts: [{ text: message }] },
  ];
  const systemInstruction = SYSTEM_INSTRUCTIONS[mode] || SYSTEM_INSTRUCTIONS.coach;

  const modelsToTry = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"];
  let lastError = null;

  for (const model of modelsToTry) {
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents,
            systemInstruction: { role: "user", parts: [{ text: systemInstruction }] },
            generationConfig: { temperature: 0.7 },
          }),
        }
      );

      if (!response.ok) {
        if (response.status === 400 && API_KEY.startsWith("gen-lang-client-")) {
          return `(Mock AI Response - Setup Needed) Hello! I received: "${message}". Please update the GEMINI_API_KEY in your .env file with a valid key from Google AI Studio to unlock real AI responses.`;
        }

        const errorText = await response.text();

        // If 404 not found, throw error to catch and try next model
        if (response.status === 404) {
          throw new Error("404");
        }

        throw new Error(`Gemini API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data: any = await response.json();
      const text = data.candidates?.[0]?.content?.parts?.map((p: any) => p.text || "").join("") || "";
      return text || "I'm sorry, I couldn't process that. Could you try again?";

    } catch (e: any) {
      if (e.message === "404") {
        lastError = "Model not found";
        continue;
      }
      throw e;
    }
  }

  throw new Error(`Gemini API error: Models not found or not supported for this API key.`);
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());
  app.use(cookieParser());

  const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 10,
    message: { error: "Too many login attempts. Try again later." },
    standardHeaders: true,
    legacyHeaders: false,
  });

  // --- PUBLIC AUTH ROUTES ---

  app.post("/api/auth/signup", async (req, res) => {
    try {
      const fullName = sanitize(req.body.full_name);
      const email = sanitize(req.body.email).toLowerCase();
      const password = req.body.password;
      const confirmPassword = req.body.confirm_password;
      const age = Number(req.body.age);
      const careerField = sanitize(req.body.career_field);
      const experienceYears = Number(req.body.experience_years) || 0;
      const currentRole = req.body.current_role != null ? sanitize(req.body.current_role) : null;
      const salaryRange = req.body.salary_range != null ? sanitize(req.body.salary_range) : null;

      if (!fullName || !email || !password) {
        return res.status(400).json({ error: "Full name, email, and password are required" });
      }
      if (password !== confirmPassword) {
        return res.status(400).json({ error: "Passwords do not match" });
      }
      const pv = validatePassword(password);
      if (!pv.ok) return res.status(400).json({ error: pv.message });
      if (!Number.isInteger(age) || age < 16 || age > 120) {
        return res.status(400).json({ error: "Age must be between 16 and 120" });
      }
      if (!careerField) return res.status(400).json({ error: "Career field is required" });

      const existing = db.prepare("SELECT id FROM users WHERE email = ?").get(email);
      if (existing) return res.status(409).json({ error: "Email already registered" });

      const id = randomUUID();
      const password_hash = await bcrypt.hash(password, SALT_ROUNDS);
      db.prepare(
        `INSERT INTO users (id, full_name, email, password_hash, age, career_field, experience_years, current_role, salary_range)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(id, fullName, email, password_hash, age, careerField, experienceYears, currentRole || null, salaryRange || null);

      const token = jwt.sign({ userId: id, email }, JWT_SECRET, { expiresIn: "7d" });
      res.cookie("token", token, COOKIE_OPTIONS);
      const user = db.prepare(
        "SELECT id, full_name, email, age, career_field, experience_years, current_role, salary_range, created_at FROM users WHERE id = ?"
      ).get(id) as any;
      res.status(201).json({ ...user, name: user.full_name });
    } catch (e) {
      console.error("Signup error:", e);
      res.status(500).json({ error: "Registration failed" });
    }
  });

  app.post("/api/auth/login", loginLimiter, async (req, res) => {
    try {
      const email = sanitize(req.body.email).toLowerCase();
      const password = req.body.password;
      if (!email || !password) {
        return res.status(400).json({ error: "Email and password are required" });
      }
      const user = db.prepare(
        "SELECT id, full_name, email, password_hash, age, career_field, experience_years, current_role, salary_range, created_at FROM users WHERE email = ?"
      ).get(email) as any;
      if (!user) return res.status(401).json({ error: "Invalid email or password" });
      const match = await bcrypt.compare(password, user.password_hash);
      if (!match) return res.status(401).json({ error: "Invalid email or password" });
      const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: "7d" });
      res.cookie("token", token, COOKIE_OPTIONS);
      delete user.password_hash;
      res.json({ ...user, name: user.full_name });
    } catch (e) {
      console.error("Login error:", e);
      res.status(500).json({ error: "Login failed" });
    }
  });

  app.post("/api/auth/logout", (_req, res) => {
    res.clearCookie("token", COOKIE_OPTIONS);
    res.json({ success: true });
  });

  app.get("/api/me", (req, res) => {
    const token = req.cookies?.token;
    if (!token) return res.status(401).json({ error: "Not authenticated" });
    try {
      const decoded = jwt.verify(token, JWT_SECRET) as JwtPayload;
      const user = db.prepare(
        "SELECT id, full_name, email, age, career_field, experience_years, current_role, salary_range, created_at FROM users WHERE id = ?"
      ).get(decoded.userId) as any;
      if (!user) return res.status(401).json({ error: "User not found" });
      res.json({ ...user, name: user.full_name });
    } catch {
      res.clearCookie("token", COOKIE_OPTIONS);
      return res.status(401).json({ error: "Invalid or expired token" });
    }
  });

  // --- PROTECTED API ROUTES ---

  app.get("/api/chats/:userId", authMiddleware, (req, res) => {
    const userId = req.params.userId;
    if (req.user!.id !== userId) return res.status(403).json({ error: "Forbidden" });
    const chats = db.prepare("SELECT * FROM chats WHERE user_id = ? ORDER BY created_at DESC").all(userId);
    res.json(chats);
  });

  app.post("/api/chats", authMiddleware, (req, res) => {
    const { title } = req.body;
    const userId = req.user!.id;
    const result = db.prepare("INSERT INTO chats (user_id, title) VALUES (?, ?)").run(userId, title || "New chat");
    res.json({ id: result.lastInsertRowid, title: title || "New chat" });
  });

  app.get("/api/messages/:chatId", authMiddleware, (req, res) => {
    const messages = db.prepare("SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC").all(req.params.chatId);
    res.json(messages);
  });

  app.post("/api/messages", authMiddleware, (req, res) => {
    const { chatId, role, content } = req.body;
    db.prepare("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)").run(chatId, role, content);
    res.json({ success: true });
  });

  app.post("/api/chat/generate", authMiddleware, async (req, res) => {
    try {
      if (!API_KEY) {
        return res.status(503).json({
          error: "AI is not configured. Add API_KEY or GEMINI_API_KEY to the server .env file and restart.",
          code: "NO_API_KEY",
        });
      }
      const { mode, history, userMessage } = req.body;
      const text = await sendMessage(userMessage, history, mode);
      res.json({ text });
    } catch (error: any) {
      console.error("AI Error:", error);
      const message = error?.message?.includes("API key")
        ? "API key missing or invalid. Check your .env file."
        : error?.message || "Failed to generate response";
      res.status(500).json({ error: message });
    }
  });

  // --- FEATURE PAGE API ROUTES (public placeholder) ---
  app.get("/api/features/ai-chat", (_req, res) => res.json({ feature: "ai-chat", title: "AI Chat", placeholder: true }));
  app.get("/api/features/find-mentor", (_req, res) => res.json({ feature: "find-mentor", title: "Find a Mentor", placeholder: true }));
  app.get("/api/features/salary-impact", (_req, res) => res.json({ feature: "salary-impact", title: "Salary Impact", placeholder: true }));
  app.get("/api/features/negotiation-practice", (_req, res) => res.json({ feature: "negotiation-practice", title: "Negotiation Practice", placeholder: true }));
  app.get("/api/features/profile", (_req, res) => res.json({ feature: "profile", title: "Profile", placeholder: true }));

  // --- VITE MIDDLEWARE ---
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: "spa" });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, "dist")));
    app.get("*", (_req, res) => res.sendFile(path.join(__dirname, "dist", "index.html")));
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`SheThrive Server running on http://localhost:${PORT}`);
    if (!API_KEY) {
      console.warn("⚠️  Gemini API key not set. Add API_KEY or GEMINI_API_KEY to .env for AI chat to work.");
    }
  });
}

startServer();
