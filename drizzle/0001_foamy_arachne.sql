CREATE TABLE `online_room_chat_messages` (
	`seq` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`room_code` text NOT NULL,
	`message_id` text NOT NULL,
	`player_id` text NOT NULL,
	`author_name` text NOT NULL,
	`body` text NOT NULL,
	`created_at` integer NOT NULL,
	FOREIGN KEY (`room_code`) REFERENCES `online_rooms`(`code`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `online_room_chat_message_id_idx` ON `online_room_chat_messages` (`room_code`,`message_id`);--> statement-breakpoint
CREATE INDEX `online_room_chat_room_seq_idx` ON `online_room_chat_messages` (`room_code`,`seq`);--> statement-breakpoint
CREATE INDEX `online_room_chat_player_time_idx` ON `online_room_chat_messages` (`room_code`,`player_id`,`created_at`);