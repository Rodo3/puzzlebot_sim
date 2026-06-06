import React from 'react';

const TEAM_MEMBERS = [
  {
    name: 'Rodrigo Pérez Zárate',
    role: 'Líder de proyecto',
    focus: 'Reconocimiento de voz / HMM',
    github: 'Rodo3',
  },
  {
    name: 'Integrante 2',
    role: 'Rol del integrante',
    focus: 'Área de enfoque',
    github: '',
  },
  {
    name: 'Integrante 3',
    role: 'Rol del integrante',
    focus: 'Área de enfoque',
    github: '',
  },
  {
    name: 'Integrante 4',
    role: 'Rol del integrante',
    focus: 'Área de enfoque',
    github: '',
  },
];

function LogoLarge() {
  return (
    <svg viewBox="0 0 96 72" width="96" height="72" aria-hidden="true">
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

function MemberCard({ member }) {
  return (
    <div className="team-card">
      <div className="team-card-avatar">
        {member.name.split(' ').map(w => w[0]).slice(0, 2).join('')}
      </div>
      <div className="team-card-info">
        <div className="team-card-name">{member.name}</div>
        <div className="team-card-role">{member.role}</div>
        <div className="team-card-focus muted">{member.focus}</div>
        {member.github && (
          <div className="team-card-github muted small">github.com/{member.github}</div>
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
          <p className="team-subtitle">AUTONOMOUS LOGISTICS</p>
          <p className="team-desc muted">
            Sistema autónomo de logística robótica basado en ROS 2 Humble.<br />
            Navegación SLAM, reconocimiento de voz offline y visión por computadora.
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
          {['ROS 2 Humble', 'Python 3.10', 'React + Vite', 'HMM (hmmlearn)', 'SLAM (slam_toolbox)',
            'OpenCV', 'WebSocket', 'Docker / WSL2'].map(t => (
            <span key={t} className="team-tech-tag">{t}</span>
          ))}
        </div>
      </div>

      <div className="team-footer-note muted small">
        Proyecto terminal — Ingeniería en Sistemas Computacionales · 2025–2026
      </div>

    </div>
  );
}
