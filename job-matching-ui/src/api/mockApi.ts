import type { CV, Job, User } from "../types";

const delay = (ms = 300) => new Promise(res => setTimeout(res, ms));
const LS = {
  users: 'mock_users',
  cvs: 'mock_cvs',
  jobs: 'mock_jobs',
  session: 'mock_session'
};

function uid(prefix = '') {
  return prefix + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

function read<T>(key: string, fallback: T): T {
  const raw = localStorage.getItem(key);
  return raw ? JSON.parse(raw) as T : fallback;
}
function write<T>(key: string, value: T) {
  localStorage.setItem(key, JSON.stringify(value));
}

export type AuthSession = { userId: string } | null;

export const api = {
  // ===== Auth =====
  async register(payload: Omit<User, 'id' | 'createdAt'>): Promise<User> {
    await delay();
    const users = read<User[]>(LS.users, []);
    if (users.some(u => u.email === payload.email)) {
      throw new Error('Email đã tồn tại');
    }
    const user: User = { id: uid('u_'), createdAt: new Date().toISOString(), ...payload };
    users.push(user);
    write(LS.users, users);
    write(LS.session, { userId: user.id } satisfies AuthSession);
    return user;
  },

  async login(email: string, password: string): Promise<User> {
    await delay();
    const users = read<User[]>(LS.users, []);
    const user = users.find(u => u.email === email && u.password === password);
    if (!user) throw new Error('Sai email hoặc mật khẩu');
    write(LS.session, { userId: user.id } satisfies AuthSession);
    return user;
  },

  async logout(): Promise<void> {
    await delay(100);
    write<AuthSession>(LS.session, null);
  },

  async currentUser(): Promise<User | null> {
    await delay(100);
    const sess = read<AuthSession>(LS.session, null);
    if (!sess) return null;
    const users = read<User[]>(LS.users, []);
    return users.find(u => u.id === sess.userId) ?? null;
  },

  // ===== CV (Candidate) =====
  async listCVsByUser(userId: string): Promise<CV[]> {
    await delay();
    const cvs = read<CV[]>(LS.cvs, []);
    return cvs.filter(c => c.userId === userId).sort((a,b) => b.updatedAt.localeCompare(a.updatedAt));
  },

  async createCV(userId: string, cv: Omit<CV, 'id' | 'userId' | 'updatedAt'>): Promise<CV> {
    await delay();
    const cvs = read<CV[]>(LS.cvs, []);
    const newCV: CV = { id: uid('cv_'), userId, updatedAt: new Date().toISOString(), ...cv };
    cvs.push(newCV);
    write(LS.cvs, cvs);
    return newCV;
  },

  async updateCV(cvId: string, patch: Partial<CV>): Promise<CV> {
    await delay();
    const cvs = read<CV[]>(LS.cvs, []);
    const idx = cvs.findIndex(c => c.id === cvId);
    if (idx < 0) throw new Error('CV không tồn tại');
    cvs[idx] = { ...cvs[idx], ...patch, updatedAt: new Date().toISOString() };
    write(LS.cvs, cvs);
    return cvs[idx];
  },

  // ===== Jobs (Employer) =====
  async listJobsByEmployer(employerId: string): Promise<Job[]> {
    await delay();
    const jobs = read<Job[]>(LS.jobs, []);
    return jobs.filter(j => j.employerId === employerId).sort((a,b)=> b.postedAt.localeCompare(a.postedAt));
  },

  async createJob(employerId: string, job: Omit<Job, 'id' | 'employerId' | 'postedAt'>): Promise<Job> {
    await delay();
    const jobs = read<Job[]>(LS.jobs, []);
    const newJob: Job = { id: uid('job_'), employerId, postedAt: new Date().toISOString(), ...job };
    jobs.push(newJob);
    write(LS.jobs, jobs);
    return newJob;
  }
};
