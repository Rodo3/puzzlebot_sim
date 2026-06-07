import React from 'react';

const TEAM_MEMBERS = [
  {
    name: 'Jesús Javier Martínez Hernández',
    studentId: 'A00833296',
    role: 'SLAM y percepción',
    github: 'jesusMBhuy',
  },
  {
    name: 'Rodolfo Alejandro Hernández Ibarra',
    studentId: 'A00828736',
    role: 'Navegación autónoma',
    github: 'Rodo3',
  },
  {
    name: 'Jorge Ignacio Reyes Pérez',
    studentId: 'A00573981',
    role: 'Dashboard y reconocimiento de voz',
    github: 'Gees14',
  },
  {
    name: 'Valeria Aranza Cerda Ochoa',
    studentId: 'A01236733',
    role: 'Navegación y localización',
    github: 'valeriaacerda',
  },
];

function LogoLarge() {
  return (
    <svg className="team-logo-mark" viewBox="0 0 96 72" aria-hidden="true">
      <path d="M6 66 L32 10 L44 34 L32 66Z" fill="#f0f0f8" />
      <path d="M32 10 L44 34 L54 18 L64 34 L76 10 L90 66 L74 66 L64 40 L54 52 L44 40 L36 66 L20 66Z"
            fill="url(#mGradLg)" />
      <path d="M18 58 L26 28" stroke="#22d3ee" strokeWidth="3" strokeLinecap="round" fill="none"/>
      <polygon points="22 24 32 20 28 34" fill="#22d3ee" />
      <circle cx="6" cy="66" r="5" fill="none" stroke="#7c3aed" strokeWidth="2.5"/>
      <circle cx="90" cy="66" r="4" fill="#22d3ee" />
      <defs>
        <linearGradient id="mGradLg" x1="32" y1="10" x2="90" y2="66" gradientUnits="userSpaceOnUse">
          <stop offset="0%"   stopColor="#7c3aed" />
          <stop offset="100%" stopColor="#a855f7" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 2C6.48 2 2 6.58 2 12.26c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.5v-1.8c-2.78.62-3.37-1.22-3.37-1.22-.45-1.19-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.34 1.12 2.91.86.09-.66.35-1.12.63-1.37-2.22-.26-4.55-1.14-4.55-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.3 9.3 0 0 1 12 7.06c.85 0 1.71.12 2.51.35 1.9-1.33 2.74-1.05 2.74-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9v2.81c0 .28.18.6.69.5A10.15 10.15 0 0 0 22 12.26C22 6.58 17.52 2 12 2Z"
      />
    </svg>
  );
}

function MemberCard({ member }) {
  return (
    <div className="team-card">
      <div className="team-card-avatar">
        {member.name.split(' ').map(w => w[0]).slice(0, 2).join('')}
      </div>
      <div className="team-card-info">
        <div className="team-card-name">{member.name}</div>
        <div className="team-card-github muted small">{member.studentId}</div>
        <div className="team-card-role">{member.role}</div>
        {member.github && (
          <a
            className="team-github-link"
            href={`https://github.com/${member.github}`}
            target="_blank"
            rel="noreferrer"
          >
            <GitHubIcon />
            <span>{member.github}</span>
          </a>
        )}
      </div>
    </div>
  );
}

export default function TeamPanel() {
  return (
    <div className="team-view">

      <div className="team-hero">
        <LogoLarge />
        <div className="team-hero-text">
          <h1 className="team-name">ANTI MASS</h1>
          <p className="team-subtitle">TE3003B · Grupo 502</p>
          <p className="team-desc muted">
            Integración de Robótica y Sistemas Inteligentes · Campus Monterrey · FJ 2026<br />
            Sistema autónomo de logística robótica basado en ROS 2 Humble.
          </p>
        </div>
      </div>

      <div className="team-section-title">Integrantes del equipo</div>

      <div className="team-grid">
        {TEAM_MEMBERS.map((m, i) => (
          <MemberCard key={i} member={m} />
        ))}
      </div>

      <div className="team-tech">
        <div className="team-section-title">Tecnologías</div>
        <div className="team-tech-tags">
          {['ROS 2 Humble', 'Python 3.10', 'React + Vite', 'HMM (hmmlearn)', 'SLAM',
            'OpenCV', 'ArUco', 'WebSocket', 'Docker / WSL2'].map(t => (
            <span key={t} className="team-tech-tag">{t}</span>
          ))}
        </div>
      </div>

      <div className="team-footer-note muted small">
        Coordinador: Prof. Alfredo Esquivel, Ph. D. · Autonomous Logistics
      </div>

    </div>
  );
}
