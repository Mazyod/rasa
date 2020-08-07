module.exports = {
  someSidebar: [
    {
      type: 'category',
      label: 'User Guide',
      collapsed: false,
      items: [
        'index',
        'user-guide/installation',
        'user-guide/training-data-format',
        'user-guide/prototype-an-assistant',
        'user-guide/building-assistants',
        'user-guide/command-line-interface',
        'user-guide/architecture',
        'user-guide/messaging-and-voice-channels',
        'user-guide/testing-your-assistant',
        'user-guide/setting-up-ci-cd',
        'user-guide/validate-files',
        'user-guide/configuring-http-api',
        'user-guide/how-to-deploy',
        'user-guide/cloud-storage',
      ],
    },
    {
      type: 'category',
      label: 'NLU',
      collapsed: false,
      items: [
        'nlu/about',
        'nlu/using-nlu-only',
        'nlu/training-data-format',
        'nlu/language-support',
        'nlu/choosing-a-pipeline',
        'nlu/components',
        'nlu/entity-extraction',
      ],
    },
    {
      type: 'category',
      label: 'Core',
      collapsed: false,
      items: [
        'core/about',
        'core/stories',
        'core/rules',
        'core/domains',
        'core/responses',
        'core/actions',
        'core/reminders-and-external-events',
        'core/policies',
        'core/slots',
        'core/forms',
        'core/retrieval-actions',
        'core/interactive-learning',
        'core/fallback-actions',
        'core/knowledge-bases',
      ],
    },
    {
      type: 'category',
      label: 'Conversation Design',
      collapsed: false,
      items: [
        'dialogue-elements/dialogue-elements',
        'dialogue-elements/small-talk',
        'dialogue-elements/completing-tasks',
        'dialogue-elements/guiding-users',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api/action-server',
        'api/http-api',
        'api/jupyter-notebooks',
        'api/agent',
        'api/custom-nlu-components',
        'api/rasa-sdk',
        'api/events',
        'api/tracker',
        'api/tracker-stores',
        'api/event-brokers',
        'api/lock-stores',
        'api/training-data-importers',
        'api/core-featurization',
        'api/tensorflow_usage',
        'migration-guide',
        'changelog',
      ],
    },
    {
      type: 'category',
      label: 'Migrate from (beta)',
      collapsed: false,
      items: [
        'migrate-from/google-dialogflow-to-rasa',
        'migrate-from/facebook-wit-ai-to-rasa',
        'migrate-from/microsoft-luis-to-rasa',
        'migrate-from/ibm-watson-to-rasa',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: ['glossary'],
    },
  ],
};
