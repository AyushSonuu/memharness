import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'memharness',
  tagline: 'Framework-agnostic memory infrastructure for AI agents',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://ayushsonuu.github.io',
  baseUrl: '/memharness/',

  organizationName: 'AyushSonuu',
  projectName: 'memharness',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/AyushSonuu/memharness/tree/main/docs/',
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/memharness-social-card.png',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'memharness',
      logo: {
        alt: 'memharness Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api/harness',
          label: 'API',
          position: 'left',
        },
        {
          href: 'https://github.com/AyushSonuu/memharness',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://pypi.org/project/memharness/',
          label: 'PyPI',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
            {
              label: 'Memory Types',
              to: '/docs/memory-types/conversational',
            },
            {
              label: 'API Reference',
              to: '/docs/api/harness',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/AyushSonuu/memharness',
            },
            {
              label: 'PyPI',
              href: 'https://pypi.org/project/memharness/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Ayush Sonuu. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'yaml', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
