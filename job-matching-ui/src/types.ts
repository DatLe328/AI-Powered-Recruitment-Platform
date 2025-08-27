export type Role = 'candidate' | 'employer';

export interface User {
  id: string;
  email: string;
  password: string; // Mock only (đừng dùng vậy ở prod)
  role: Role;
  fullName?: string;
  companyName?: string;
  createdAt: string;
}

export interface CVExperience {
  company: string;
  role: string;
  years: number; // có thể là số năm 1.5, 2, ...
  description?: string;
}

export interface CV {
  id: string;
  userId: string;
  title: string;
  summary: string;
  skills: string[];
  experience: CVExperience[];
  fileBase64?: string; // nếu upload file (PDF/DOCX) → demo
  updatedAt: string;
}

export type EmploymentType = 'full-time' | 'part-time' | 'contract' | 'intern';

export interface Job {
  id: string;
  employerId: string; // userId của employer
  title: string;
  description: string;
  skills: string[];
  salaryMin?: number;
  salaryMax?: number;
  location: string;
  employmentType: EmploymentType;
  postedAt: string;
}
